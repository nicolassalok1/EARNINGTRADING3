import sys
import io
import math
import random
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from scipy import stats
from scipy.optimize import minimize

BASE_DIR = Path(__file__).parent
TO_COPY = BASE_DIR / "to_copy"
NEXT_DIR = BASE_DIR / "next"
if str(TO_COPY) not in sys.path:
    sys.path.insert(0, str(TO_COPY))

from finance import Finance
from dqlagent_pytorch import DQLAgent, device
from bsm73 import bsm_call_value
from assetallocation_pytorch import Investing, InvestingAgent


# ---------- Helpers ----------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def capture_logs(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return buf.getvalue(), result


@st.cache_data(show_spinner=False)
def load_raw_data():
    url = "https://certificate.tpq.io/rl4finance.csv"
    return pd.read_csv(url, index_col=0, parse_dates=True).dropna()


@st.cache_data(show_spinner=False)
def load_next_csv(name, parse_dates=None):
    path = NEXT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Fichier manquant : {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def train_test_split_series(series, train_ratio=0.8):
    n_train = max(5, int(len(series) * train_ratio))
    train = series.iloc[:n_train]
    test = series.iloc[n_train:]
    return train, test


def polyfit_predict(series, deg=2):
    x = np.arange(len(series))
    coeffs = np.polyfit(x, series.values, deg=deg)
    poly = np.poly1d(coeffs)
    preds = poly(x)
    next_pred = float(poly(len(series)))
    return preds, next_pred


def inverse_variance_weights(cov):
    inv_var = 1 / np.clip(np.diag(cov), 1e-8, None)
    w = inv_var / inv_var.sum()
    return w


def pca_svd(matrix, n_components=3):
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    comps = Vt[:n_components]
    explained = (s ** 2) / (len(matrix) - 1)
    explained_ratio = explained / explained.sum()
    scores = np.dot(matrix, comps.T)
    return comps, scores, explained_ratio


def bsm_put_value(St, K, T, t, r, sigma):
    call = bsm_call_value(St, K, T, t, r, sigma)
    return call - St + K * math.exp(-r * (T - t))


def plot_rewards(rewards, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(rewards, color="tab:blue", lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.grid(True, alpha=0.3)
    return fig


# ---------- Hedging classes (notebook 07) ----------
class ObservationSpace:
    def __init__(self, n):
        self.shape = (n,)


class ActionSpace:
    def __init__(self, n):
        self.n = n

    def seed(self, seed):
        random.seed(seed)

    def sample(self):
        return random.random()


def simulate_gbm(S0, T, r, sigma, steps):
    path = [S0]
    dt = T / steps
    for _ in range(steps):
        st = path[-1] * math.exp((r - sigma ** 2 / 2) * dt +
                                 sigma * math.sqrt(dt) * random.gauss(0, 1))
        path.append(st)
    return np.array(path)


def bsm_delta(St, K, T, t, r, sigma):
    d1 = ((math.log(St / K) + (r + 0.5 * sigma ** 2) * (T - t)) /
          (sigma * math.sqrt(T - t)))
    return stats.norm.cdf(d1, 0, 1)


def option_replication(path, K, T, r, sigma):
    dt = T / (len(path) - 1)
    bond = [math.exp(r * i * dt) for i in range(len(path))]
    res = pd.DataFrame()
    for i in range(len(path) - 1):
        C = bsm_call_value(path[i], K, T, i * dt, r, sigma)
        if i == 0:
            s = bsm_delta(path[i], K, T, i * dt, r, sigma)
            b = (C - s * path[i]) / bond[i]
        else:
            V = s * path[i] + b * bond[i]
            s = bsm_delta(path[i], K, T, i * dt, r, sigma)
            b = (C - s * path[i]) / bond[i]
            df = pd.DataFrame({"St": path[i], "C": C, "V": V,
                               "s": s, "b": b}, index=[0])
            res = pd.concat((res, df), ignore_index=True)
    return res


class Hedging:
    def __init__(self, S0, K_, T, r_, sigma_, steps):
        self.initial_value = S0
        self.strike_ = K_
        self.maturity = T
        self.short_rate_ = r_
        self.volatility_ = sigma_
        self.steps = steps
        self.observation_space = ObservationSpace(5)
        self.osn = self.observation_space.shape[0]
        self.action_space = ActionSpace(1)
        self._simulate_data()
        self.portfolios = pd.DataFrame()
        self.episode = 0

    def _simulate_data(self):
        s = [self.initial_value]
        self.strike = random.choice(self.strike_)
        self.short_rate = random.choice(self.short_rate_)
        self.volatility = random.choice(self.volatility_)
        self.dt = self.maturity / self.steps
        for _ in range(1, self.steps + 1):
            st = s[-1] * math.exp(
                (self.short_rate - self.volatility ** 2 / 2) * self.dt +
                self.volatility * math.sqrt(self.dt) * random.gauss(0, 1))
            s.append(st)
        self.data = pd.DataFrame(s, columns=["index"])
        self.data["bond"] = np.exp(self.short_rate *
                                   np.arange(len(self.data)) * self.dt)

    def _get_state(self):
        St = self.data["index"].iloc[self.bar]
        Bt = self.data["bond"].iloc[self.bar]
        ttm = self.maturity - self.bar * self.dt
        if ttm > 0:
            Ct = bsm_call_value(St, self.strike, self.maturity,
                                self.bar * self.dt, self.short_rate,
                                self.volatility)
        else:
            Ct = max(St - self.strike, 0)
        return np.array([St, Bt, ttm, Ct, self.strike, self.short_rate,
                         self.stock, self.bond]), {}

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def reset(self):
        self.bar = 0
        self.bond = 0
        self.stock = 0
        self.treward = 0
        self.episode += 1
        self._simulate_data()
        self.state, _ = self._get_state()
        return self.state, _

    def step(self, action):
        if self.bar == 0:
            reward = 0
            self.bar += 1
            self.stock = float(action)
            self.bond = ((self.state[3] - self.stock * self.state[0]) /
                         self.state[1])
            self.new_state, _ = self._get_state()
        else:
            self.bar += 1
            self.new_state, _ = self._get_state()
            phi_value = (self.stock * self.new_state[0] +
                         self.bond * self.new_state[1])
            pl = phi_value - self.new_state[3]
            df = pd.DataFrame({"e": self.episode, "s": self.stock,
                               "b": self.bond, "phi": phi_value,
                               "C": self.new_state[3], "p&l[$]": pl,
                               "p&l[%]": pl / max(self.new_state[3], 1e-4) * 100,
                               "St": self.new_state[0],
                               "Bt": self.new_state[1],
                               "K": self.strike, "r": self.short_rate,
                               "sigma": self.volatility}, index=[0])
            self.portfolios = pd.concat((self.portfolios, df),
                                        ignore_index=True)
            reward = -(phi_value - self.new_state[3]) ** 2
            self.stock = float(action)
            self.bond = ((self.new_state[3] -
                          self.stock * self.new_state[0]) /
                         self.new_state[1])
        done = self.bar == len(self.data) - 1
        self.state = self.new_state
        return self.state, float(reward), done, False, {}


class HedgingAgent(DQLAgent):
    def opt_action(self, state):
        bnds = [(0, 1)]

        def f_obj(x):
            s = state.copy()
            s[0, 6] = x
            s[0, 7] = ((s[0, 3] - x * s[0, 0]) / s[0, 1])
            s_tensor = torch.FloatTensor(s).to(device)
            with torch.no_grad():
                q_val = self.model(s_tensor)
            return q_val.cpu().numpy()[0, 0]

        try:
            res = minimize(lambda x: -f_obj(x), 0.5,
                           bounds=bnds, method="Powell")
            action = res["x"][0]
        except Exception:
            action = self.env.stock
        return action

    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        return self.opt_action(state)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            target = torch.tensor([reward], dtype=torch.float32).to(device)
            if not done:
                ns = next_state.copy()
                opt_act = self.opt_action(ns)
                ns[0, 6] = opt_act
                ns[0, 7] = ((ns[0, 3] - opt_act * ns[0, 0]) / ns[0, 1])
                ns_tensor = torch.FloatTensor(ns).to(device)
                with torch.no_grad():
                    future_q = self.model(ns_tensor)[0, 0]
                target = target + self.gamma * future_q
            state_tensor = torch.FloatTensor(state).to(device)
            self.optimizer.zero_grad()
            current_q = self.model(state_tensor)[0, 0]
            loss = self.criterion(current_q, target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def test(self, episodes, verbose=True):
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = self._reshape(state)
            treward = 0
            for _ in range(1, len(self.env.data) + 1):
                action = self.opt_action(state)
                state, reward, done, trunc, _ = self.env.step(action)
                state = self._reshape(state)
                treward += reward
                if done:
                    if verbose:
                        templ = f"total penalty={treward:4.2f}"
                        print(templ)
                    break


# ---------- Tabs ----------
st.set_page_config(page_title="RL4F Notebooks en Streamlit", layout="wide")
st.title("RL4F : trois notebooks regroupés")
st.caption("Chaque onglet reflète un notebook (06, 07, 08) avec des paramètres ajustables.")

tabs = st.tabs([
    "06 · Trading DQL",
    "07 · Hedging",
    "08 · Allocation 3AC",
    "Next · Stock Pred",
    "Next · Derivatives Pricing",
    "Next · Derivatives Hedging",
    "Next · BTC Strategy",
    "Next · BTC PCA",
    "Next · Portfolio Alloc",
    "Next · Eigen Portfolio",
    "Next · YC Build",
    "Next · YC Predict",
])


with tabs[0]:
    st.subheader("Notebook 06 — Agent DQL sur données historiques")
    st.write("Entraînement rapide d’un agent DQL sur la série `rl4finance.csv` "
             "via l’environnement `Finance`.")

    data = load_raw_data()
    symbols = sorted(data.columns)
    col_a, col_b, col_c = st.columns([1.5, 1, 1])
    with col_a:
        symbol = st.selectbox("Symbole", symbols, index=0)
    with col_b:
        n_features = st.slider("Fenêtre (lags)", 4, 30, 8, 1)
        min_acc = st.slider("Seuil d'exactitude", 0.0, 1.0, 0.50, 0.01)
    with col_c:
        episodes = st.slider("Episodes d'entraînement", 1, 200, 20, 1)
        test_episodes = st.slider("Episodes de test", 1, 50, 5, 1)
    seed = st.number_input("Seed", value=100, step=1)
    lr = st.number_input("Learning rate", value=0.0005, format="%.6f")

    st.dataframe(data[[symbol]].head(), use_container_width=True)

    if st.button("Lancer l'entraînement Trading"):
        set_global_seed(seed)
        finance = Finance(symbol, "r", min_accuracy=min_acc,
                          n_features=n_features)
        agent = DQLAgent(finance.symbol, finance.feature,
                         finance.n_features, finance, lr=lr)
        with st.spinner("Entraînement en cours..."):
            train_logs, _ = capture_logs(agent.learn, episodes)
        finance.min_accuracy = 0.0
        with st.spinner("Tests..."):
            test_logs, _ = capture_logs(agent.test, test_episodes,
                                        min_accuracy=0.0,
                                        min_performance=0.0,
                                        verbose=False, full=False)
        st.success("Terminé.")
        st.text_area("Journal (entrainement + test)",
                     (train_logs + test_logs)[-4000:], height=180)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Episodes vus", len(agent.trewards))
        with col2:
            st.metric("Max reward", f"{agent.max_treward:.2f}")
        with col3:
            st.metric("Epsilon final", f"{agent.epsilon:.3f}")

        if agent.trewards:
            fig = plot_rewards(agent.trewards,
                               "Récompense cumulée par épisode")
            st.pyplot(fig)


with tabs[1]:
    st.subheader("Notebook 07 — Hedging (delta et DQL)")
    st.write("Simulation GBM + réplication delta et agent DQL pour ajuster la couverture.")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        S0 = st.number_input("S0", value=100.0, step=1.0)
        T = st.number_input("Maturité (années)", value=1.0, step=0.25)
    with col_b:
        K = st.number_input("Strike K", value=100.0, step=5.0)
        steps = st.slider("Pas de temps", 50, 400, 252, 10)
    with col_c:
        r = st.number_input("Taux r", value=0.05, format="%.4f")
        sigma = st.number_input("Vol (sigma)", value=0.20, format="%.4f")
    with col_d:
        seed_h = st.number_input("Seed hedging", value=750, step=1)
        episodes_h = st.slider("Episodes DQL", 1, 100, 10, 1)
        test_h = st.slider("Episodes test", 1, 50, 5, 1)

    st.markdown("**Réplication delta BSM**")
    if st.button("Simuler réplication delta"):
        set_global_seed(seed_h)
        path = simulate_gbm(S0, T, r, sigma, steps)
        rep = option_replication(path, K, T, r, sigma)
        st.write(rep.head())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(path, lw=1.2, color="tab:blue")
        ax1.set_title("Trajectoire simulée")
        ax1.set_xlabel("Pas")
        ax1.set_ylabel("Prix")
        if not rep.empty:
            ax2.plot(rep["C"].values, label="Call", lw=1.1)
            ax2.plot(rep["V"].values, label="Portefeuille", lw=1.1)
            ax2.set_title("Call vs portefeuille répliquant")
            ax2.legend()
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.markdown("**Agent de couverture DQL**")
    lr_h = st.number_input("Learning rate (hedging)", value=0.001, format="%.6f")
    hu_h = st.slider("Hidden units", 8, 256, 128, 8)
    if st.button("Entraîner l'agent Hedging"):
        set_global_seed(seed_h)
        Ks = np.array([0.9, 0.95, 1.0, 1.05, 1.10]) * K
        rs = [0, r / 2, r]
        sigmas = [max(sigma / 2, 1e-4), sigma, sigma * 1.5]
        env = Hedging(S0=S0, K_=Ks, T=T, r_=rs, sigma_=sigmas, steps=steps)
        agent = HedgingAgent("SYM", feature=None, n_features=8,
                             env=env, hu=hu_h, lr=lr_h)
        with st.spinner("Entraînement..."):
            train_logs, _ = capture_logs(agent.learn, episodes_h)
        env.portfolios = pd.DataFrame()
        with st.spinner("Tests..."):
            test_logs, _ = capture_logs(agent.test, test_h, verbose=False)
        st.success("Terminé.")
        st.text_area("Journal hedging", (train_logs + test_logs)[-4000:],
                     height=160)
        st.metric("Epsilon final", f"{agent.epsilon:.3f}")
        if not env.portfolios.empty:
            last_ep = env.portfolios["e"].max()
            sample = env.portfolios[env.portfolios["e"] == last_ep]
            st.write("P&L échantillon (dernier épisode)")
            st.dataframe(sample[["p&l[$]", "p&l[%]", "St", "C"]].head(),
                         use_container_width=True)
            fig = plot_rewards(agent.trewards, "Pénalité cumulée")
            st.pyplot(fig)


with tabs[2]:
    st.subheader("Notebook 08 — Allocation à trois actifs (InvestingAgent)")
    st.write("Réplique l’agent d’allocation 3AC avec un entraînement court.")

    df = load_raw_data()
    symbols = sorted(df.columns)
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        asset_x = st.selectbox("Actif X", symbols, index=0)
    with col_b:
        asset_y = st.selectbox("Actif Y", symbols, index=1)
    with col_c:
        asset_z = st.selectbox("Actif Z", symbols, index=2)
    with col_d:
        steps_inv = st.slider("Pas (jours)", 50, 400, 252, 10)
        seed_inv = st.number_input("Seed allocation", value=100, step=1)
    episodes_inv = st.slider("Episodes entraînement", 1, 100, 10, 1)
    test_inv = st.slider("Episodes test", 1, 50, 10, 1)
    lr_inv = st.number_input("Learning rate allocation", value=0.00025,
                             format="%.6f")
    hu_inv = st.slider("Hidden units allocation", 8, 256, 128, 8)

    st.dataframe(df[[asset_x, asset_y, asset_z]].head(),
                 use_container_width=True)

    if st.button("Lancer l'agent d'allocation"):
        set_global_seed(seed_inv)
        investing = Investing(asset_x, asset_y, asset_z,
                              steps=steps_inv, amount=1)
        agent = InvestingAgent("3AC", feature=None, n_features=6,
                               env=investing, hu=hu_inv, lr=lr_inv)
        with st.spinner("Entraînement allocation..."):
            train_logs, _ = capture_logs(agent.learn, episodes_inv)
        investing.portfolios = pd.DataFrame()
        with st.spinner("Tests allocation..."):
            test_logs, _ = capture_logs(agent.test, test_inv, verbose=False)
        st.success("Terminé.")
        st.text_area("Journal allocation", (train_logs + test_logs)[-4000:],
                     height=160)
        if not investing.portfolios.empty:
            last_ep = investing.portfolios["e"].max()
            sample = investing.portfolios[investing.portfolios["e"] == last_ep]
            alloc_cols = ["xt", "yt", "zt", "pv"]
            st.write("Trajectoire (dernier épisode)")
            st.dataframe(sample[alloc_cols].head(), use_container_width=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(sample["pv"].values, lw=1.4, color="tab:green")
            ax.set_title("Valeur de portefeuille")
            ax.set_xlabel("Pas")
            ax.set_ylabel("PV")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)


with tabs[3]:
    st.subheader("Next · Stock Return Prediction")
    st.write("Régression simple (polyfit) sur la série `data.csv`.")

    df = load_next_csv("data.csv", parse_dates=["Date"])
    df = df.set_index("Date")
    feature = st.selectbox("Colonne cible", ["Close", "Adj Close", "Open", "High", "Low"])
    degree = st.slider("Degré du polynôme", 1, 3, 2)
    train_ratio = st.slider("Taille train (%)", 60, 95, 80, 1) / 100

    series = df[feature].dropna()
    train, test = train_test_split_series(series, train_ratio)
    preds_train, next_pred = polyfit_predict(train, deg=degree)
    x_all = np.arange(len(series))
    poly = np.poly1d(np.polyfit(np.arange(len(train)), train.values, deg=degree))
    preds_all = poly(x_all)
    test_pred = preds_all[len(train):len(series)]
    mae = float(np.mean(np.abs(test.values - test_pred))) if len(test) else float("nan")

    st.metric("MAE (test)", f"{mae:.2f}")
    st.metric("Prévision prochain point", f"{next_pred:.2f}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(series.index, series.values, label="Réel", lw=1.5)
    ax.plot(series.index[:len(train)], preds_train, label="Fit train", lw=1.2)
    if len(test_pred):
        ax.plot(series.index[len(train):], test_pred, label="Préd test", lw=1.2, linestyle="--")
    ax.legend()
    ax.set_title(f"Prédiction sur {feature}")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


with tabs[4]:
    st.subheader("Next · Derivatives Pricing")
    st.write("Prix BSM (call/put) et courbe en fonction du strike.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        S = st.number_input("Spot S", value=100.0, step=1.0)
        K = st.number_input("Strike K", value=100.0, step=1.0)
    with col_b:
        T = st.number_input("Maturité (années)", value=1.0, step=0.25)
        r = st.number_input("Taux r", value=0.01, format="%.4f")
    with col_c:
        sigma = st.number_input("Vol (sigma)", value=0.20, format="%.4f")
        step_strikes = st.slider("Variation autour de K", 5, 50, 20, 5)

    call = bsm_call_value(S, K, T, 0, r, sigma)
    put = bsm_put_value(S, K, T, 0, r, sigma)
    st.metric("Call BSM", f"{call:.4f}")
    st.metric("Put BSM", f"{put:.4f}")

    strikes = np.linspace(max(1, K - step_strikes), K + step_strikes, 25)
    call_curve = [bsm_call_value(S, k, T, 0, r, sigma) for k in strikes]
    put_curve = [bsm_put_value(S, k, T, 0, r, sigma) for k in strikes]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(strikes, call_curve, label="Call", lw=1.4)
    ax.plot(strikes, put_curve, label="Put", lw=1.4)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Prix")
    ax.set_title("Courbes de prix BSM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


with tabs[5]:
    st.subheader("Next · Derivatives Hedging")
    st.write("Réplication delta avec GBM simulé (version rapide).")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        S0_h = st.number_input("S0", value=100.0, step=1.0, key="dh_s0")
        K_h = st.number_input("Strike", value=100.0, step=1.0, key="dh_k")
    with col_b:
        T_h = st.number_input("Maturité", value=1.0, step=0.25, key="dh_t")
        r_h = st.number_input("Taux", value=0.01, format="%.4f", key="dh_r")
    with col_c:
        sigma_h = st.number_input("Vol", value=0.2, format="%.4f", key="dh_sigma")
        steps_h = st.slider("Steps", 30, 300, 120, 10, key="dh_steps")
    seed_dh = st.number_input("Seed", value=123, step=1, key="dh_seed")

    if st.button("Simuler réplication (Next)"):
        set_global_seed(seed_dh)
        path = simulate_gbm(S0_h, T_h, r_h, sigma_h, steps_h)
        rep = option_replication(path, K_h, T_h, r_h, sigma_h)
        st.write(rep.head())
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(path, lw=1.1, label="Sous-jacent")
        ax2 = ax.twinx()
        ax2.plot(rep["C"].values, color="tab:red", label="Call", lw=1.1)
        ax2.plot(rep["V"].values, color="tab:green", label="Portefeuille", lw=1.1, linestyle="--")
        ax.set_title("Réplication delta")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        st.pyplot(fig)


with tabs[6]:
    st.subheader("Next · Bitcoin Trading Strategy")
    st.write("Stratégie simple de croisement de moyennes mobiles sur `BitstampData.csv`.")

    df_btc = load_next_csv("BitstampData.csv")
    df_btc = df_btc.dropna()
    df_btc["Date"] = pd.to_datetime(df_btc["Timestamp"], unit="s")
    df_btc = df_btc.set_index("Date").sort_index()
    short = st.slider("SMA courte", 3, 50, 10, 1)
    long = st.slider("SMA longue", 10, 200, 50, 5)
    df_btc["ret"] = df_btc["Close"].pct_change()
    df_btc["sma_s"] = df_btc["Close"].rolling(short).mean()
    df_btc["sma_l"] = df_btc["Close"].rolling(long).mean()
    df_btc["signal"] = np.sign(df_btc["sma_s"] - df_btc["sma_l"])
    df_btc["strat_ret"] = df_btc["signal"].shift(1) * df_btc["ret"]
    perf = (1 + df_btc[["ret", "strat_ret"]].dropna()).cumprod()
    st.metric("Perf stratégie (x)", f"{perf['strat_ret'].iloc[-1]:.3f}")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(perf.index, perf["ret"], label="Buy & Hold", lw=1.0)
    ax.plot(perf.index, perf["strat_ret"], label="Stratégie SMA", lw=1.2)
    ax.legend()
    ax.set_title("Cumul des rendements")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


with tabs[7]:
    st.subheader("Next · Bitcoin PCA (réduction de dimension)")
    st.write("PCA rapide sur OHLCV pour visualiser les composantes principales.")

    df_btc_pca = load_next_csv("BitstampData.csv")
    df_btc_pca = df_btc_pca.dropna().copy()
    df_btc_pca["Date"] = pd.to_datetime(df_btc_pca["Timestamp"], unit="s")
    df_btc_pca = df_btc_pca.set_index("Date").sort_index()
    feats = ["Open", "High", "Low", "Close", "Volume_(BTC)"]
    X = df_btc_pca[feats].values
    comps, scores, explained = pca_svd(X, n_components=3)
    st.write("Variance expliquée", [f"{x:.2%}" for x in explained])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(scores[:, 0], scores[:, 1], s=6, alpha=0.4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Projection PCA (PC1 vs PC2)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


with tabs[8]:
    st.subheader("Next · Portfolio Allocation (crypto)")
    st.write("Pondérations inverse-variance ou égalité sur `crypto_portfolio.csv`.")

    df_crypto = load_next_csv("crypto_portfolio.csv", parse_dates=["Date"])
    df_crypto = df_crypto.set_index("Date")
    ret = df_crypto.pct_change().dropna()
    method = st.radio("Méthode de pondération", ["Égal pondéré", "Inverse variance"])
    cov = ret.cov()
    if method == "Inverse variance":
        w = inverse_variance_weights(cov.values)
    else:
        w = np.repeat(1 / ret.shape[1], ret.shape[1])
    weights = pd.Series(w, index=ret.columns)
    st.write("Pondérations", weights.round(4))
    strat = (ret * weights).sum(axis=1)
    perf = (1 + strat).cumprod()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(perf.index, perf.values, lw=1.3, color="tab:purple")
    ax.set_title("Valeur de portefeuille (base 1)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


with tabs[9]:
    st.subheader("Next · Eigen Portfolio (Dow 30)")
    st.write("Première composante propre du covariance Dow_adjcloses.")

    df_dow = load_next_csv("Dow_adjcloses.csv", parse_dates=["Date"]).set_index("Date")
    ret = df_dow.pct_change().dropna()
    cov = ret.cov()
    vals, vecs = np.linalg.eigh(cov.values)
    idx = vals.argsort()[::-1]
    top_vec = vecs[:, idx[0]]
    weights = top_vec / np.sum(np.abs(top_vec))
    w_series = pd.Series(weights, index=ret.columns)
    st.write("Pondérations (normalisées)", w_series.round(4).sort_values(ascending=False))
    strat = (ret * w_series).sum(axis=1)
    perf = (1 + strat).cumprod()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(perf.index, perf.values, lw=1.2, color="tab:orange")
    ax.set_title("Eigen-portfolio (PC1)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


with tabs[10]:
    st.subheader("Next · Yield Curve Construction")
    st.write("PCA sur courbe de swap (`DownloadedData.csv`).")

    yc = load_next_csv("DownloadedData.csv", parse_dates=["DATE"]).set_index("DATE")
    comps, scores, explained = pca_svd(yc.values, n_components=3)
    st.write("Variance expliquée", [f"{x:.2%}" for x in explained])
    k = st.slider("Nombre de composantes pour reconstruction", 1, 3, 2)
    recon = scores[:, :k] @ comps[:k]
    recon += yc.values.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(yc.columns, yc.iloc[-1].values, label="Dernière courbe", lw=1.4)
    ax.plot(yc.columns, recon[-1], label=f"Reconstruction {k} PC", lw=1.2, linestyle="--")
    ax.set_title("Courbe de swap")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


with tabs[11]:
    st.subheader("Next · Yield Curve Prediction")
    st.write("Projection d’un pas en avant par lissage exponentiel sur `DownloadedData.csv`.")

    yc = load_next_csv("DownloadedData.csv", parse_dates=["DATE"]).set_index("DATE")
    alpha = st.slider("Facteur de lissage", 0.01, 0.99, 0.2, 0.01)
    forecast = yc.ewm(alpha=alpha).mean().iloc[-1]
    st.write("Prévision prochaine courbe")
    st.dataframe(forecast.to_frame("Prévision").T, use_container_width=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(yc.columns, yc.iloc[-1].values, label="Dernière observation", lw=1.2)
    ax.plot(yc.columns, forecast.values, label="Prévision", lw=1.2, linestyle="--")
    ax.set_title("Prévision courte échéance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
