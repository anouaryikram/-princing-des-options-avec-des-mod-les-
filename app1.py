%%writefile app.py
# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

st.set_page_config(layout="wide", page_title="Pricing d'une option européenne")

# ----------------- STYLE -----------------
st.markdown(
    """
    <style>
    .main { background-color: #e9eefc; }
    .title-box { background-color: #e9eefc; padding: 18px; text-align: center; }
    .left-box, .right-box, .small-box {
        background-color: #eef3ff;
        padding: 10px;
        border-radius: 4px;
        border: 3px solid #ffffff;
    }
    .greek-box {
        background-color: #eef3ff;
        padding: 6px;
        border-radius: 6px;
        border: 2px solid #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- FONCTIONS FINANCIÈRES -----------------
def black_scholes_price(S, K, T, r, sigma, option="call"):
    if T <= 0:
        return float(max(S - K, 0.0) if option == "call" else max(K - S, 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(price)

def black_scholes_greeks(S, K, T, r, sigma, option="call"):
    if T <= 0:
        return {"Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    Delta = norm.cdf(d1) if option == "call" else norm.cdf(d1) - 1
    Gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    Vega = S * pdf_d1 * np.sqrt(T)
    if option == "call":
        Theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        Rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        Theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        Rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return {
        "Delta": float(Delta),
        "Gamma": float(Gamma),
        "Vega": float(Vega / 100.0),
        "Theta": float(Theta / 365.0),
        "Rho": float(Rho / 100.0)
    }

def crr_price(S, K, T, r, sigma, steps=50, option="call", return_tree=False):
    if T <= 0:
        return (max(S-K,0) if option=="call" else max(K-S,0))
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    a = math.exp(r * dt)
    p = (a - d) / (u - d)
    # prix à maturité
    ST = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    # backward induction
    for i in range(steps - 1, -1, -1):
        payoff = np.array([math.exp(-r * dt) * (p * payoff[j + 1] + (1 - p) * payoff[j]) for j in range(i + 1)])
    price = float(payoff[0])
    if return_tree:
        tree = []
        for i in range(steps + 1):
            row = [S * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
            tree.append(row)
        return price, tree
    return price

def trinomial_price(S, K, T, r, sigma, steps=3, option="call", return_tree=False):
    # Trinomial recombinaison (simple implémentation)
    if T <= 0:
        return (max(S-K,0) if option=="call" else max(K-S,0))
    dt = T / steps
    # paramètre de saut (choix classique pour trinomial)
    u = math.exp(sigma * math.sqrt(2 * dt))
    d = 1 / u
    # probabilités approximées (une construction symétrique)
    # on évite les divisions par zéro en cas de sigma très faible
    denom = (math.exp(sigma * math.sqrt(dt / 2)) - math.exp(-sigma * math.sqrt(dt / 2)))
    if denom == 0:
        # fallback à CRR si sigma ~ 0
        return crr_price(S, K, T, r, sigma, steps, option)
    pu = ((math.exp(r * dt / 2) - math.exp(-sigma * math.sqrt(dt / 2))) / denom) ** 2 / 2
    pd = ((math.exp(sigma * math.sqrt(dt / 2)) - math.exp(r * dt / 2)) / denom) ** 2 / 2
    pm = 1 - pu - pd
    # construire l'arbre des prix (chaque ligne a 2*i+1 nœuds)
    tree = []
    for i in range(steps + 1):
        row = []
        for j in range(2 * i + 1):
            # position j correspond à un déplacement net (i - j)
            S_ij = S * (u ** (i - j)) * (d ** j)
            row.append(S_ij)
        tree.append(row)
    # payoffs à la maturité
    values = []
    for S_T in tree[-1]:
        values.append(max(S_T - K, 0) if option == "call" else max(K - S_T, 0))
    # remontée arrière
    for i in range(steps - 1, -1, -1):
        new_values = []
        # note: values correspond à 2*(i+1)+1 éléments ; pour node j on combine values[j], values[j+1], values[j+2]
        for j in range(len(tree[i])):
            v = math.exp(-r * dt) * (pu * values[j] + pm * values[j + 1] + pd * values[j + 2])
            new_values.append(v)
        values = new_values
    price = values[0]
    if return_tree:
        return price, tree
    return price

def monte_carlo_price(S, K, T, r, sigma, n_paths=20000, option="call", seed=0):
    if T <= 0:
        return (max(S-K,0) if option=="call" else max(K-S,0))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * Z)
    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    price = math.exp(-r * T) * float(np.mean(payoff))
    return price

# ----------------- INTERFACE -----------------
st.markdown("<div class='title-box'><h1>Pricing d'une option européenne</h1></div>", unsafe_allow_html=True)

left_col, center_col, right_col = st.columns([1, 3, 1])

# --- paramètres gauche ---
with left_col:
    st.markdown("<div class='left-box'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:bold'>Paramètres</div>", unsafe_allow_html=True)
    T = st.number_input("T (maturité en années)", value=0.5, min_value=0.0, step=0.01, format="%.4f")
    K = st.number_input("K (strike)", value=100.0, step=1.0)
    S0 = st.number_input("So (prix actuel sous-jacent)", value=100.0, step=1.0)
    r = st.number_input("r (taux sans risque)", value=0.05, format="%.4f")
    sigma = st.number_input("Volatilité (sigma)", value=0.15, format="%.4f")
    option_type = st.selectbox("Call/Put", options=["Call", "Put"])
    st.button("Valider")
    st.markdown("</div>", unsafe_allow_html=True)

# --- choix modèles droite ---
with right_col:
    st.markdown("<div class='right-box'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:bold'>Choix des modèles</div>", unsafe_allow_html=True)
    sel_crr = st.checkbox("CRR", value=False)
    sel_bs = st.checkbox("Black Scholes", value=False)
    sel_mc = st.checkbox("Monte Carlo", value=False)
    sel_tri = st.checkbox("Trinomial", value=False)
    st.markdown("---")
    steps = st.number_input("Périodes (pour modèles binomiaux/trinomial)", min_value=1, value=3, step=1)
    mc_paths = st.number_input("Simulations Monte Carlo", min_value=100, value=20000, step=100)
    st.markdown("</div>", unsafe_allow_html=True)

option = option_type.lower()

# ----------------- SECTION PRIX (2x2) -----------------
with center_col:
    st.markdown("<div style='background-color:#eef3ff; padding:12px; border:4px solid white; border-radius:6px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center'>Prix</h2>", unsafe_allow_html=True)
    # --- Vérification d'arbitrage ---
    if sigma < 0:
        st.markdown(
            """
            <div style='color:red; font-weight:bold; text-align:center; font-size:20px;'>
            Erreur : Arbitrage<br>
            (la volatilité ne peut pas être négative)
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()  # on arrête l'exécution ici pour éviter les calculs incohérents

    fig, axs = plt.subplots(2, 2, figsize=(10,8))
    axs = axs.flatten()

    # subplot 0: CRR (arbre)
    ax0 = axs[0]
    if sel_crr:
        price_crr, tree_crr = crr_price(S0, K, T, r, sigma, steps=steps, option=option, return_tree=True)
        for i, row in enumerate(tree_crr):
            xs = np.full(len(row), i)
            ax0.scatter(xs, row)
            for j, val in enumerate(row):
                ax0.text(i, val, f"{val:.2f}", fontsize=7, ha='center', va='bottom')
            if i > 0:
                prev = tree_crr[i-1]
                # chaque noeud prev[p] connecte à row[p] et row[p+1]
                for j in range(len(row)):
                    parent_indices = [j-1, j] if j-1 >= 0 else [j]
                    for p in parent_indices:
                        if 0 <= p < len(prev):
                            ax0.plot([i-1, i], [prev[p], row[j]], color='blue')
        ax0.set_title("Modèle Binomial (CRR)")
        ax0.set_xlabel("Étape")
        ax0.set_ylabel("Prix")
    else:
        ax0.set_visible(False)

    # subplot 1: Black & Scholes (prix vs maturité)
    ax1 = axs[1]
    if sel_bs:
        ts = np.linspace(0.0001, T if T > 0 else 0.5, 50)
        vals_bs = [black_scholes_price(S0, K, t, r, sigma, option) for t in ts]
        ax1.plot(ts, vals_bs, marker='o')
        ax1.set_title("Black & Scholes")
        ax1.set_xlabel("Maturité")
        ax1.set_ylabel("Prix")
    else:
        ax1.set_visible(False)

    # subplot 2: Monte Carlo (prix vs maturité)
    ax2 = axs[2]
    if sel_mc:
        ts2 = np.linspace(0.0001, T if T > 0 else 0.5, 6)
        # réduire nb de chemins pour les évaluations intermédiaires pour accélérer
        mc_eval_paths = int(max(200, min(mc_paths, int(mc_paths / 4))))
        vals_mc = [monte_carlo_price(S0, K, t, r, sigma, n_paths=mc_eval_paths, option=option, seed=42) for t in ts2]
        ax2.plot(ts2, vals_mc, marker='o')
        ax2.set_title("Monte Carlo (prix vs maturité)")
        ax2.set_xlabel("Maturité")
        ax2.set_ylabel("Prix")
    else:
        ax2.set_visible(False)

    # subplot 3: Trinomial (arbre)
    ax3 = axs[3]
    if sel_tri:
        price_tri, tree_tri = trinomial_price(S0, K, T, r, sigma, steps=steps, option=option, return_tree=True)
        for i, row in enumerate(tree_tri):
            xs = np.full(len(row), i)
            ax3.scatter(xs, row)
            for j, val in enumerate(row):
                ax3.text(i, val, f"{val:.2f}", fontsize=7, ha='center', va='bottom')
            if i > 0:
                prev = tree_tri[i-1]
                # pour chaque parent prev[p] connecter aux trois enfants row[p], row[p+1], row[p+2]
                for p in range(len(prev)):
                    if p < len(row):
                        ax3.plot([i-1, i], [prev[p], row[p]], color='blue')
                    if p+1 < len(row):
                        ax3.plot([i-1, i], [prev[p], row[p+1]], color='blue')
                    if p+2 < len(row):
                        ax3.plot([i-1, i], [prev[p], row[p+2]], color='blue')
        ax3.set_title("Modèle Trinomial")
        ax3.set_xlabel("Étape")
        ax3.set_ylabel("Prix")
    else:
        ax3.set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- SECTION EVOLUTION (graphe combiné) -----------------
with left_col:
    st.markdown("<div style='margin-top:16px; background-color:#eef3ff; padding:12px; border-radius:6px; border:3px solid white;'>", unsafe_allow_html=True)
    st.markdown("<b>Évolution des prix (maturité)</b>", unsafe_allow_html=True)
    fig2 = plt.figure(figsize=(6.5,4.0))
    ax_e = fig2.add_subplot(1,1,1)

    t_vals = np.linspace(0.0001, T if T > 0 else 0.5, 8)

    plotted = False
    if sel_crr:
        vals_crr = [crr_price(S0, K, t, r, sigma, steps=steps, option=option) for t in t_vals]
        ax_e.plot(t_vals, vals_crr, label="CRR", marker='o')
        plotted = True
    if sel_bs:
        vals_bs = [black_scholes_price(S0, K, t, r, sigma, option) for t in t_vals]
        ax_e.plot(t_vals, vals_bs, label="Black & Scholes", marker='o')
        plotted = True
    if sel_mc:
        mc_eval_paths = int(max(200, min(mc_paths, int(mc_paths / 4))))
        vals_mc = [monte_carlo_price(S0, K, t, r, sigma, n_paths=mc_eval_paths, option=option, seed=42) for t in t_vals]
        ax_e.plot(t_vals, vals_mc, label="Monte Carlo", marker='o')
        plotted = True
    if sel_tri:
        vals_tri = [trinomial_price(S0, K, t, r, sigma, steps=steps, option=option) for t in t_vals]
        ax_e.plot(t_vals, vals_tri, label="Trinomial", marker='o')
        plotted = True

    if plotted:
        ax_e.set_xlabel("Maturité")
        ax_e.set_ylabel("Prix")
        ax_e.legend()
    else:
        ax_e.text(0.5, 0.5, "Aucun modèle sélectionné", ha='center', va='center')
        ax_e.set_xticks([])
        ax_e.set_yticks([])

    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Affichage des prix numériques et greeks à droite -----------------
with right_col:
    st.markdown("<div class='right-box' style='padding-top:10px'>", unsafe_allow_html=True)

    # Bouton de calcul
    calculer = st.button("🧮 Calculer les prix", key="calc_prices")

    # Placeholders dynamiques (vides au départ)
    crr_placeholder = st.empty()
    bs_placeholder = st.empty()
    mc_placeholder = st.empty()
    tri_placeholder = st.empty()

    if calculer:
        # Calcul des prix uniquement après clic
        if sel_crr:
            price_crr = crr_price(S0, K, T, r, sigma, steps=steps, option=option)
            crr_placeholder.markdown(f"Prix CRR : {price_crr:.6f}")

        if sel_bs:
            price_bs = black_scholes_price(S0, K, max(T, 1e-9), r, sigma, option)
            bs_placeholder.markdown(f"Prix Black-Scholes : {price_bs:.6f}")

        if sel_mc:
            price_mc = monte_carlo_price(S0, K, T, r, sigma, n_paths=int(mc_paths), option=option, seed=42)
            mc_placeholder.markdown(f"Prix Monte Carlo : {price_mc:.6f}")

        if sel_tri:
            price_tri = trinomial_price(S0, K, T, r, sigma, steps=steps, option=option)
            tri_placeholder.markdown(f"Prix Trinomial : {price_tri:.6f}")

    else:
        st.info("🟡 Clique sur « Calculer les prix » pour afficher les résultats.")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Boîte des Greeks dynamiques ---
    st.markdown("<div class='greek-box'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:bold; text-align:center;'>Greeks</div>", unsafe_allow_html=True)

    if calculer and sel_bs:
        greeks_bs = black_scholes_greeks(S0, K, max(T,1e-9), r, sigma, option)
        st.markdown("<u>Black-Scholes</u>", unsafe_allow_html=True)
        for g, v in greeks_bs.items():
            st.markdown(f"{g}** : {v:.6f}")
    else:
        st.write("Les Greeks s’affichent après calcul Black-Scholes.")

    st.markdown("</div>", unsafe_allow_html=True)