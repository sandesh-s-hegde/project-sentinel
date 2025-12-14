\# 👁️ Project Sentinel



\### Empirically Grounding Operational Resilience via Computer Vision



\*\*Project Sentinel\*\* is a technical proof-of-concept (PoC) designed to bridge the gap between \*\*Computer Vision\*\* and \*\*Supply Chain Finance\*\*. It addresses the "Input Paradox" in Real Options Analysis by extracting dynamic volatility metrics directly from unstructured shop-floor video feeds.



\### 🎯 Objective

Traditional Real Options models rely on "Assumed Volatility" (static historical proxies). This project deploys a \*\*"Digital Eye"\*\* (YOLOv8 + Flow Entropy) to measure \*\*"True Volatility"\*\*, allowing for the dynamic pricing of operational flexibility in real-time.



\### ⚙️ Technical Architecture

1\.  \*\*Vision Engine:\*\* YOLOv8 (Object Detection) + ByteTrack logic.

2\.  \*\*Metric:\*\* Shannon Entropy of flow vectors ($\\sigma\_{flow}$) serves as the proxy for operational chaos.

3\.  \*\*Valuation:\*\* Black-Scholes Model dynamically updated with $\\sigma\_{flow}$.



\### 🚀 Quick Start

1\.  \*\*Clone the repository:\*\*

    ```bash

    git clone \[https://github.com/YOUR\_USERNAME/project-sentinel.git](https://github.com/YOUR\_USERNAME/project-sentinel.git)

    ```

2\.  \*\*Install dependencies:\*\*

    ```bash

    pip install -r requirements.txt

    ```

3\.  \*\*Run the Dashboard:\*\*

    ```bash

    streamlit run app.py

    ```



\### 🛠️ Built With

\* \*\*Ultralytics YOLOv8\*\*: SOTA Object Detection.

\* \*\*OpenCV\*\*: Video Processing.

\* \*\*Streamlit\*\*: Interactive Dashboarding.

\* \*\*SciPy\*\*: Financial Calculus.

