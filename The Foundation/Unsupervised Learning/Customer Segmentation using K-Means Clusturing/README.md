# Mall Customer Segmentation 👥

Ever wonder why some mall ads feel perfectly targeted and others... just don't? Welcome to the magic of customer segmentation. This project is a deep dive into the `Mall_Customers.csv` dataset to figure out *who* shops at the mall and *what* they *actually* want.

But this isn't your average 2-feature K-Means project. We went a little further.

## What's Inside?

We don't just make clusters; we build **Customer Personas** 💡.

To do this, we moved beyond the basic "Income vs. Spending" plot and used a much richer, 4-dimensional approach:

  * **Smarter Features:** We use `Age`, `Gender`, `Annual Income`, AND `Spending Score` to get a complete picture.
  * **Proper Prep:** K-Means is picky about scales, so we used `StandardScaler` to put all our features on a level playing field.
  * **Better Math:** We didn't just "eyeball" the Elbow Method. We used the **Silhouette Score** to programmatically *prove* what the optimal number of clusters (`k`) should be.
  * **Actionable Insights:** The final output isn't just a colorful graph. It's a profile of each cluster's average customer, telling a business *who* these people are and *how* to market to them.

## The Tech Stack 📊

  * **Python 3.x**
  * **Pandas & Numpy:** For wrangling all that data.
  * **Scikit-learn:** The brain behind the `KMeans` clustering, `StandardScaler`, and `Silhouette Score`.
  * **Matplotlib & Seaborn:** For classic, beautiful plots.
  * **Plotly:** To render that *fancy* 3D interactive scatter plot (because our 4D analysis has a 3D soul).

## How to Run This

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/nandan2003/ML-Learning.git
    cd ML-Learning/"The Foundation"/"Unsupervised Learning"/"Customer Segmentation using K-Means Clusturing"
    ```
2.  **Install the goods:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn plotly
    ```
3.  **Run the Notebook:**
    Open `customer_segmentation_using_k_means_clustering.ipynb` in Jupyter Lab or VS Code.
4.  Run all the cells and explore the interactive 3D plot\!

## The Big Reveal: Our Customer Personas

Spoiler alert: we found (and named) 5 key groups. You'll have to run the notebook to see the full breakdown, but here's a taste:

  * **The VIPs:** (High Income, High Spending) - *The group every store wants.*
  * **The Careful Savers:** (High Income, Low Spending) - *Got the money, but won't spend it easily.*
  * **The Enthusiasts:** (Low Income, High Spending) - *Maybe students or young professionals? Great target for "buy now, pay later."*
  
  
Dive in, check out the code, and see what other insights you can find.