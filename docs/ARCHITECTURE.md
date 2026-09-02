# System Architecture & Technical Deep-Dive

This document provides a comprehensive technical overview of the architecture, algorithmic formulations, and design decisions behind **`real-estate-spatial`**.

---

## 1. System Architecture & Data Flow

`real-estate-spatial` is structured as a decoupled, modular system composed of six core subsystems:
1. **Data Ingestion & Cleaning Engine**
2. **Geospatial Core & Indexing Layer**
3. **Appraisal Comparables Engine**
4. **Machine Learning Model Suite**
5. **Valuation Pipeline Orchestrator**
6. **Submarket & Portfolio Analytics Suite**

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Data Ingestion & Cleaning                         │
 │     PropertyDataLoader (JSON / GeoJSON / CSV) ───► DataValidator       │
 │                                                           │            │
 │                                                           ▼            │
 │                                                      DataCleaner       │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Validated Properties
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                        Geospatial Indexing Core                        │
 │        ┌─────────────────────────┴─────────────────────────┐           │
 │        ▼                                                   ▼           │
 │   KDTreeIndex (Unit-Sphere 3D)                       GridIndex         │
 │        │                                                   │           │
 │        └─────────────────────────┬─────────────────────────┘           │
 │                                  ▼                                     │
 │                      SpatialFeatureExtractor                           │
 │           (Spatial Lag, Local Density, Landmark Proximity)             │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
 ┌───────────────────────────┐                     ┌───────────────────────┐
 │ Appraisal Comps Engine    │                     │ ML Valuation Engine   │
 │ - SimilarityScorer        │                     │ - Hedonic Ridge (L2)  │
 │ - CompAdjustmentCalculator│                     │ - Spatial GBM         │
 │   (GLA, Beds, Age, Trend) │                     │ - Ensemble (Inv-MAE)  │
 └─────────────┬─────────────┘                     └───────────┬───────────┘
               │                                               │
               └───────────────────────┬───────────────────────┘
                                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         Valuation Orchestrator                         │
 │   Strategy Engine: 'auto' | 'comps' | 'ensemble' | 'hedonic' | 'gbm'   │
 │   Confidence Estimation, Spread Intervals & Explanations (Drivers)     │
 └─────────────────────────────────────┬──────────────────────────────────┘
                                       │
           ┌───────────────────────────┴───────────────────────────┐
           ▼                                                       ▼
 ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
 │ Submarket Micro-Market Analytics  │   │ Portfolio Risk & Stress Testing   │
 │ - Spatial Grid & Radial Clusters  │   │ - Value Dispersion & High Risk    │
 │ - Pricing Volatility & Metrics    │   │ - Macro & Spatial Shock Scenarios │
 └───────────────────────────────────┘   └───────────────────────────────────┘
```

---

## 2. Geospatial Core & Coordinate Calculations

### 2.1 Spherical Geodesics (Haversine Formula)

Distance calculations on Earth's surface use the great-circle Haversine formula with mean Earth radius $R = 6371.0088\text{ km}$:

$$\Delta\phi = \text{lat}_2 - \text{lat}_1, \quad \Delta\lambda = \text{lon}_2 - \text{lon}_1$$

$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\text{lat}_1)\cos(\text{lat}_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)$$

$$c = 2 \cdot \operatorname{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right), \quad d = R \cdot c$$

Both scalar (`haversine_distance`) and vectorized NumPy implementations (`haversine_distance_vectorized`) are provided to accelerate batch neighborhood queries over large arrays.

### 2.2 KDTreeIndex (3D Spherical Projection)

To avoid distortion caused by flat Cartesian $(x, y)$ projections across varying latitudes, `KDTreeIndex` maps geographic coordinates $(\phi, \lambda)$ into 3D Cartesian coordinates on a unit sphere:

$$x = \cos(\phi)\cos(\lambda), \quad y = \cos(\phi)\sin(\lambda), \quad z = \sin(\phi)$$

Euclidean distance in 3D chord space $d_{3\text{D}}$ relates directly to spherical angular distance $c$ by:

$$d_{3\text{D}} = 2 \sin\left(\frac{c}{2}\right) \implies c = 2 \arcsin\left(\frac{d_{3\text{D}}}{2}\right), \quad d = R \cdot c$$

This mapping guarantees exact spatial indexing using standard $k$-d trees (either `scipy.spatial.KDTree` or pure NumPy recursive $k$-d tree fallback) for:
- $k$-Nearest Neighbors ($k$-NN)
- Radius neighborhood filtering ($r \le d_{\max}$)
- Bounding-box spatial queries

### 2.3 GridIndex

`GridIndex` offers an $O(1)$ spatial hash-map partitioning the globe into fixed angular degree cells (default $\Delta = 0.05^\circ$). Queries compute intersecting grid cells directly, minimizing distance calculations for dense urban datasets.

### 2.4 Spatial Feature Engineering

`SpatialFeatureExtractor` enriches structural property features with contextual spatial signals:

1. **Spatial Lag Feature**: Inverse-distance weighted average of neighboring property attributes (e.g. price per sqft):
   $$\operatorname{Lag}(p) = \frac{\sum_{i \in N(p)} w_i \cdot x_i}{\sum_{i \in N(p)} w_i}, \quad w_i = \frac{1}{(d(p, p_i) + \epsilon)^p}$$
2. **Local Spatial Density**: Property count within a local radius $r$, normalized by search area $\pi r^2$.
3. **Landmark Proximity**: Great-circle distances and directional offsets to key regional landmarks (CBDs, transit hubs, coastlines).

---

## 3. Valuation Algorithms Deep-Dive

### 3.1 Automated Appraisal Comparables Engine

The `CompsEngine` identifies top comparable properties and applies formal appraisal adjustments:

#### Multi-Attribute Similarity Scoring (`SimilarityScorer`)
A composite similarity score $S(p_{\text{subj}}, p_{\text{comp}}) \in [0, 1]$ is computed as a weighted linear combination:

$$S = w_{\text{dist}} S_{\text{dist}} + w_{\text{sqft}} S_{\text{sqft}} + w_{\text{beds}} S_{\text{beds}} + w_{\text{baths}} S_{\text{baths}} + w_{\text{age}} S_{\text{age}} + w_{\text{lot}} S_{\text{lot}} + w_{\text{type}} S_{\text{type}}$$

- **Spatial Distance Penalty**: $S_{\text{dist}} = \exp\left(-\frac{d}{d_{\text{decay}}}\right)$
- **Continuous Metric Similarities**: $S_x = \max\left(0, 1 - \frac{|x_{\text{subj}} - x_{\text{comp}}|}{\sigma_x}\right)$
- **Categorical Property Type Match**: $S_{\text{type}} = 1.0 \text{ if } \text{type}_{\text{subj}} == \text{type}_{\text{comp}} \text{ else } 0.0$

#### Comp Monetary Adjustments (`CompAdjustmentCalculator`)
Line-item adjustments reconcile differences between the comparable property and the subject property:

1. **Gross Living Area (GLA / Sqft)**:
   $$\Delta_{\text{sqft}} = (\text{sqft}_{\text{subj}} - \text{sqft}_{\text{comp}}) \times \text{PPSQFT}_{\text{comp}} \times \alpha_{\text{marginal}}$$
2. **Bedrooms & Bathrooms**:
   $$\Delta_{\text{bed}} = (\text{beds}_{\text{subj}} - \text{beds}_{\text{comp}}) \times V_{\text{bed}}$$
   $$\Delta_{\text{bath}} = (\text{baths}_{\text{subj}} - \text{baths}_{\text{comp}}) \times V_{\text{bath}}$$
3. **Lot Size**:
   $$\Delta_{\text{lot}} = (\text{lot}_{\text{subj}} - \text{lot}_{\text{comp}}) \times V_{\text{lot\_sqft}}$$
4. **Age / Effective Year Built**:
   $$\Delta_{\text{age}} = (\text{year\_built}_{\text{subj}} - \text{year\_built}_{\text{comp}}) \times \beta_{\text{deprec}} \times \text{Price}_{\text{comp}}$$
5. **Market Time Trend Appreciation**:
   $$\Delta_{\text{time}} = \text{Price}_{\text{comp}} \times \left((1 + r_{\text{annual}})^{\Delta t / 365.25} - 1\right)$$

The final adjusted price is:
$$\text{Price}_{\text{adj}} = \text{Price}_{\text{comp}} + \sum \Delta_k$$

The weighted comps valuation combines adjusted prices weighted by similarity scores:
$$\hat{V}_{\text{comps}} = \frac{\sum_i S_i \cdot \text{Price}_{\text{adj}, i}}{\sum_i S_i}$$

---

### 3.2 Hedonic Spatial Ridge Regression

`HedonicSpatialRegression` models property pricing as a function of structural variables, spatial coordinates, and coordinate interaction polynomials.

#### Feature Matrix
$$X = \left[ 1, \quad \text{sqft}, \quad \text{beds}, \quad \text{baths}, \quad \text{lot}, \quad \text{age}, \quad \phi, \quad \lambda, \quad \phi^2, \quad \lambda^2, \quad \phi \cdot \lambda \right]$$

#### Closed-Form Solver
Parameters $\beta$ are estimated via L2-regularized Ridge regression:

$$\hat{\beta} = \left(X^T X + \alpha I\right)^{-1} X^T y$$

This provides fast training and inference with exact closed-form matrix operations.

---

### 3.3 Spatial Gradient Boosting (`SpatialGradientBoostingModel`)

Combines structural property features with extracted spatial lag and local density features.

- **Primary Engine**: Uses `xgboost.XGBRegressor` when installed.
- **Pure NumPy Fallback**: Implements an exact gradient boosted tree ensemble with gradient descent on squared error loss when XGBoost is not present.

---

### 3.4 Ensemble Valuation Model (`EnsembleValuationModel`)

Combines multiple underlying models (Hedonic Ridge, Spatial GBM, and Comps) into a consolidated prediction:

$$\hat{V}_{\text{ensemble}} = \sum_{m} w_m \hat{V}_m, \quad \sum_m w_m = 1.0$$

#### Weighting Strategies
1. **Performance-Based (Inverse-MAE)**:
   $$w_m \propto \frac{1}{\operatorname{MAE}_m + \epsilon}$$
2. **Equal Weighting**:
   $$w_m = \frac{1}{M}$$
3. **Custom Weights**: User-specified fixed weight allocations.

---

## 4. Valuation Pipeline & Orchestrator

The `ValuationOrchestrator` unifies data validation, cleaning, indexing, model execution, and explainability into a single workflow.

### Strategy Fallback Logic (`strategy="auto"`)

```
              ┌──────────────────────────────────────────────┐
              │          Input Subject Property              │
              └──────────────────────┬───────────────────────┘
                                     │
                                     ▼
                      Are >= 3 Comps Available &
                      Average Similarity >= 0.65?
                     /                         \
                   YES                          NO
                   /                             \
     Is ML Ensemble Fitted?               Is ML Ensemble Fitted?
         /            \                       /            \
       YES             NO                   YES             NO
       /                \                   /                \
 Blended Valuation    Comps Valuation   Ensemble Only   Heuristic Fallback
 (Comps + Ensemble)      Estimate         Valuation        (PPSQFT median)
```

### Confidence Score Derivation
Confidence $C \in [0.1, 0.99]$ combines:
- Comparable quality and density ($S_{\text{avg}}$ and count $k$).
- Spatial distance to training distribution.
- Model consensus / divergence between sub-models.

### Confidence Bounds & Spread
$$\text{Lower Bound} = \hat{V} \times (1 - (1 - C) \times 0.5)$$
$$\text{Upper Bound} = \hat{V} \times (1 + (1 - C) \times 0.5)$$
$$\text{Spread \%} = \frac{\text{Upper Bound} - \text{Lower Bound}}{\hat{V}} \times 100$$

---

## 5. Submarket & Portfolio Analytics

### 5.1 Submarket Micro-Market Analyzer (`SubmarketAnalyzer`)
- Partitions large datasets into micro-markets using spatial grid binning or radial distance clustering.
- Calculates median price, mean price per sqft, and price volatility index $\sigma_P / \mu_P$.

### 5.2 Portfolio Risk & Stress Assessor (`PortfolioAssessor`)
- **Risk Identification**: Flags properties with confidence below threshold ($C < C_{\text{risk}}$) or valuation spread exceeding limits ($\text{Spread} > \tau_{\text{spread}}$).
- **Macroeconomic & Spatial Stress Testing**:
  Applies macroeconomic shocks ($\Delta_{\text{macro}}\%$) alongside localized spatial circular shock hotspots ($(\phi_h, \lambda_h, r_h, \Delta_{\text{local}}\%)$):

$$\text{Price}_{\text{stressed}}(p) = \hat{V}(p) \times (1 + \Delta_{\text{macro}}) \times \prod_{h \in H, d(p, c_h) \le r_h} (1 + \Delta_{\text{local}, h})$$
