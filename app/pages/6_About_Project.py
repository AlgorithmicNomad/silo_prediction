import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")

st.title("ℹ️ Project Information")

st.markdown("""
## Physics-Informed Machine Learning System for Reinforced Concrete Silo Stress Prediction

This platform demonstrates the synergy between **Classical Structural Engineering** and **Modern Artificial Intelligence**.

### 📖 Technical Background

1. **Janssen's Silo Theory (1895)**: 
   Unlike shallow bins, deep silos exhibit a phenomenon where material pressure does not increase linearly with depth. Instead, wall friction carries a significant portion of the vertical load. This is mathematically captured by Janssen's equation.

2. **Synthetic Engineering Datasets**: 
   In many structural engineering scenarios, real-world failure data is scarce. This project utilizes physics-informed synthetic data generation to create robust training sets that obey physical laws while incorporating realistic measurement noise.

3. **Machine Learning Strategy**: 
   We employ high-performance regression algorithms (XGBoost, Random Forest) to learn the mapping from geometric parameters (Height, Diameter, Thickness) and material properties to internal structural stresses.

---

### 📐 Governing Equations

**Lateral Pressure ($p_z$):**
$$p(z) = \frac{\gamma D}{4 \mu K} \left(1 - e^{\frac{-4 \mu K z}{D}}\right)$$

**Hoop Stress ($\sigma_h$):**
$$\sigma_h = \frac{p(z) D}{2t}$$

**Variables:**
- $z$: Depth from the top of the material
- $\gamma$: Bulk density of the stored material
- $D$: Internal diameter of the silo
- $\mu$: Coefficient of friction between material and wall
- $K$: Ratio of horizontal to vertical pressure
- $t$: Concrete wall thickness

---

### 🚀 Future Roadmap
- **Finite Element Analysis (FEA)** integration for complex 3D stress distributions.
- **Dynamic Loading**: Analysis of seismic and wind impacts.
- **Deep Learning**: Implementing Multi-Layer Perceptrons (MLP) for more complex material behaviors.
- **Auto-Design**: An AI agent that recommends wall thickness based on safety factors.

### 🏛️ References
- Janssen, H. A., "Versuche über Getreidedruck in Silozellen," Zeitschrift des Vereines Deutscher Ingenieure, 1895.
- Eurocode 1: Actions on structures - Part 4: Silos and tanks.
""")
