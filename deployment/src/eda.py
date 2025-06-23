import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image

def run():
    # membuat title
    st.title('Aplikasi Prediksi Cancer Severity')

    # membuat sub header
    st.subheader('Page ini mengenai EDA dari Dataset Cancer Severity')

    # menampilkan dataframe
    df = pd.read_csv(r'./src/global_cancer_patients_2015_2024.csv')
    st.dataframe(df)

    # ubah nama column
    df.columns = [col.lower() for col in df.columns]
    
    # eda 1
    st.write("### Average Treatment Cost (USD) by Country/Region")
    country_treatment_cost = df.groupby('country_region')['treatment_cost_usd'].mean()
    fig = plt.figure(figsize=(10, 6))
    country_treatment_cost.sort_values().plot(kind='barh', color='skyblue')
    plt.title("Average Treatment Cost (USD) by Country/Region")
    plt.xlabel("Treatment Cost (USD)")
    plt.ylabel("Country/Region")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

    # eda 2
    st.write("### Histogram berdasarkan input user")
    option = st.selectbox('Pilih Kolom:', ('cancer_type', 'cancer_stage'))
    fig1 = plt.figure(figsize=(15, 5))
    sns.countplot(x=option, data=df, order=df[option].value_counts().index)
    plt.title(f'Countplot of {option}')
    plt.xlabel(option)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig1)

    # eda 3
    st.write("### Cancer Type Count by Gender")
    gender_cancer_type = df.groupby('gender')['cancer_type'].value_counts()
    gender_cancer_type_df = gender_cancer_type.unstack().fillna(0)
    fig2 = plt.figure(figsize=(14, 6))
    ax = fig2.add_subplot(1, 1, 1)  # Tell it where to plot
    gender_cancer_type_df.plot(kind='bar', ax=ax)  # Attach plot to the right axes
    ax.set_title("Cancer Type Count by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title="Cancer Type")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig2)

    # eda 4
    st.write("### Boxplot of Numerical Features")
    cols_to_plot = df.select_dtypes(include='number').drop(columns=['treatment_cost_usd', 'target_severity_score', 'year', 'age'])
    fig3 = plt.figure(figsize=(12, 6))
    ax = fig3.add_subplot(1, 1, 1)  # Add axes to the figure
    cols_to_plot.plot(kind='box', ax=ax)  # Plot on the correct axes
    ax.set_title('Boxplot of Numerical Features')
    ax.set_ylabel('Value')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig3)

    # eda summary
    st.markdown("---")
    st.subheader("🧠 Rangkuman Hasil EDA")

    st.markdown("""
    ### 📌 Average Treatment Cost (USD) by Country
    Data menunjukkan **tidak adanya perbedaan signifikan** dalam rata-rata biaya pengobatan antar negara. Padahal seharusnya, berdasarkan data publik (misalnya dari Google), terdapat **perbedaan yang jelas** dalam biaya pengobatan kanker antara negara seperti UK, India, USA, dll.

    ### 📌 Cancer Stage Value
    Distribusi value pada fitur `cancer_stage` terlihat **terlalu merata**, yang tidak mencerminkan kondisi nyata. Biasanya distribusi stadium kanker cenderung tidak seimbang, tergantung pada prevalensi dan deteksi dini.

    ### 📌 Cancer Type by Gender
    Data menunjukkan **distribusi jenis kanker berdasarkan gender yang terlalu seimbang**. Contohnya, di dunia nyata rasio pria:wanita untuk breast cancer adalah sekitar **1:106**, namun pada dataset terlihat **jumlah pria dan wanita hampir sama**. Hal ini menimbulkan keraguan akan keakuratan distribusi gender pada jenis kanker tertentu.

    ### 📌 Data Distribution (Boxplot)
    Distribusi data numerik terlihat sangat **simetris**, memiliki **rentang nilai seragam**, dan **minim outlier**. Kemungkinan besar dataset ini telah melalui proses **pembersihan dan normalisasi** sebelum dibagikan. Hal ini juga didukung oleh pola distribusi data lainnya yang terlihat terlalu "rapi".
    """)


if __name__ == '__main__':
    run()