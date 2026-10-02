import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes, load_digits
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ==========================================
# تنظیمات اولیه
# ==========================================
st.set_page_config(page_title="پروژه جامع یادگیری ماشین", layout="wide", page_icon="🤖")

# مدیریت وضعیت برای جابجایی بین صفحات
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

# ==========================================
# ۱. صفحه اصلی (Home) - ۶ دکمه
# ==========================================
def show_home():
    st.title("🚀 سامانه جامع یادگیری ماشین")
    st.write("لطفاً یک الگوریتم را برای شروع انتخاب کنید:")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📈 رگرسیون")
        if st.button("ورود به بخش رگرسیون", use_container_width=True):
            st.session_state.page = 'menu_regression'
            st.rerun()
            
    with col2:
        st.subheader("📊 کلاسیفیکیشن")
        if st.button("ورود به بخش کلاسیفیکیشن", use_container_width=True):
            st.session_state.page = 'menu_classification'
            st.rerun()
            
    with col3:
        st.subheader("🔵 کلاستریشن")
        if st.button("ورود به بخش کلاستریشن", use_container_width=True):
            st.session_state.page = 'menu_clustering'
            st.rerun()

# ==========================================
# ۲. منوی رگرسیون (۵ دیتاست)
# ==========================================
def show_menu_regression():
    st.title("📈 دیتاست‌های رگرسیون")
    st.write("یک دیتاست را انتخاب کنید:")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("۱. دیابت (Diabetes)", use_container_width=True):
            st.session_state.dataset = 'diabetes'
            st.session_state.page = 'predict_regression'
            st.rerun()
        if st.button("۲. قیمت خانه (California)", use_container_width=True):
            st.session_state.dataset = 'california'
            st.session_state.page = 'predict_regression'
            st.rerun()
        if st.button("۳. مصرف سوخت خودرو (MPG)", use_container_width=True):
            st.info("به زودی...")
    with col2:
        if st.button("۴. نمرات دانش‌آموزان", use_container_width=True):
            st.info("به زودی...")
        if st.button("۵. فروش فروشگاه", use_container_width=True):
            st.info("به زودی...")

    if st.button("⬅️ بازگشت به منوی اصلی"):
        st.session_state.page = 'home'
        st.rerun()

# ==========================================
# ۳. منوی کلاسیفیکیشن (۵ دیتاست)
# ==========================================
def show_menu_classification():
    st.title("📊 دیتاست‌های کلاسیفیکیشن")
    st.write("یک دیتاست را انتخاب کنید:")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("۱. تایتانیک (Titanic)", use_container_width=True):
            st.session_state.dataset = 'titanic'
            st.session_state.page = 'predict_classification'
            st.rerun()
        if st.button("۲. گل‌ها (Iris)", use_container_width=True):
            st.session_state.dataset = 'iris'
            st.session_state.page = 'predict_classification'
            st.rerun()
        if st.button("۳. سرطان سینه (Breast Cancer)", use_container_width=True):
            st.session_state.dataset = 'cancer'
            st.session_state.page = 'predict_classification'
            st.rerun()
    with col2:
        if st.button("۴. شراب (Wine)", use_container_width=True):
            st.session_state.dataset = 'wine'
            st.session_state.page = 'predict_classification'
            st.rerun()
        if st.button("۵. اعداد دست‌نویس (Digits)", use_container_width=True):
            st.info("به زودی...")

    if st.button("⬅️ بازگشت به منوی اصلی"):
        st.session_state.page = 'home'
        st.rerun()

# ==========================================
# ۴. منوی کلاستریشن (۵ دیتاست)
# ==========================================
def show_menu_clustering():
    st.title("🔵 دیتاست‌های کلاستریشن")
    st.write("یک دیتاست را انتخاب کنید:")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("۱. مشتریان فروشگاه (Mall)", use_container_width=True):
            st.info("به زودی...")
        if st.button("۲. گل‌ها (Iris Clustering)", use_container_width=True):
            st.session_state.dataset = 'iris_cluster'
            st.session_state.page = 'predict_clustering'
            st.rerun()
    with col2:
        if st.button("۳. شراب (Wine Clustering)", use_container_width=True):
            st.session_state.dataset = 'wine_cluster'
            st.session_state.page = 'predict_clustering'
            st.rerun()
        if st.button("۴. داده‌های مصنوعی (Blobs)", use_container_width=True):
            st.info("به زودی...")

    if st.button("⬅️ بازگشت به منوی اصلی"):
        st.session_state.page = 'home'
        st.rerun()

# ==========================================
# ۵. صفحه پیش‌بینی رگرسیون (Diabetes)
# ==========================================
def show_predict_regression():
    st.title("🔮 پیش‌بینی با مدل رگرسیون")
    
    if st.session_state.dataset == 'diabetes':
        st.write("پیش‌بینی پیشرفت بیماری دیابت بر اساس ویژگی‌های بیمار")
        # لود دیتاست داخلی
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target
        model = LinearRegression()
        model.fit(X, y)
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("سن (نرمال‌شده):", value=0.05, format="%.3f")
            sex = st.number_input("جنسیت (نرمال‌شده):", value=0.05, format="%.3f")
            bmi = st.number_input("شاخص توده بدنی (BMI):", value=0.06, format="%.3f")
        with col2:
            bp = st.number_input("فشار خون (BP):", value=0.04, format="%.3f")
            s1 = st.number_input("کلسترول (S1):", value=0.02, format="%.3f")
            s2 = st.number_input("S2:", value=0.03, format="%.3f")
            
        if st.button("🚀 محاسبه پیش‌بینی", type="primary"):
            # ساخت ورودی (بقیه ویژگی‌ها را صفر می‌گذاریم)
            user_input = np.zeros((1, 10))
            user_input[0, 0] = age
            user_input[0, 1] = sex
            user_input[0, 2] = bmi
            user_input[0, 3] = bp
            user_input[0, 4] = s1
            user_input[0, 5] = s2
            
            prediction = model.predict(user_input)
            st.success(f"🎯 پیش‌بینی پیشرفت دیابت: {prediction[0]:.2f}")

    elif st.session_state.dataset == 'california':
        st.info("این بخش در حال ساخت است. از دیتاست Diabetes استفاده کنید.")

    if st.button("⬅️ بازگشت به لیست دیتاست‌ها"):
        st.session_state.page = 'menu_regression'
        st.rerun()

# ==========================================
# ۶. صفحه پیش‌بینی کلاسیفیکیشن (Titanic, Iris, Cancer, Wine)
# ==========================================
def show_predict_classification():
    st.title("🔮 پیش‌بینی با مدل کلاسیفیکیشن")
    
    # ------------------ تایتانیک ------------------
    if st.session_state.dataset == 'titanic':
        st.write("پیش‌بینی زنده ماندن در تایتانیک")
        try:
            df = pd.read_csv('datasets/titanic.csv')
        except FileNotFoundError:
            st.error("❌ فایل titanic.csv پیدا نشد! لطفاً آن را در پوشه datasets قرار دهید.")
            return

        df['Age'] = df['Age'].fillna(df['Age'].mean())
        df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        features = ['Pclass', 'Sex', 'Age', 'Fare']
        X = df[features]
        y = df['Survived']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("کلاس بلیط:", [1, 2, 3], index=2)
            sex = st.selectbox("جنسیت:", ["مرد", "زن"])
        with col2:
            age = st.number_input("سن:", min_value=0, max_value=100, value=30)
            fare = st.number_input("کرایه بلیط:", min_value=0.0, value=32.0)
        
        if st.button("🚀 محاسبه پیش‌بینی", type="primary"):
            sex_numeric = 0 if sex == "مرد" else 1
            user_input = pd.DataFrame([[pclass, sex_numeric, age, fare]], columns=features)
            prediction = model.predict(user_input)
            prob = model.predict_proba(user_input)[0][1]
            
            if prediction[0] == 1:
                st.success(f"🎯 زنده می‌ماند! (احتمال: {prob*100:.1f}%)")
            else:
                st.error(f"🎯 زنده نمی‌ماند. (احتمال زنده ماندن: {prob*100:.1f}%)")

    # ------------------ گل Iris ------------------
    elif st.session_state.dataset == 'iris':
        st.write("تشخیص نوع گل Iris بر اساس اندازه‌هایش")
        iris = load_iris()
        X = iris.data
        y = iris.target
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        col1, col2 = st.columns(2)
        with col1:
            sl = st.number_input("طول کاسبرگ (cm):", value=5.1)
            sw = st.number_input("عرض کاسبرگ (cm):", value=3.5)
        with col2:
            pl = st.number_input("طول گلبرگ (cm):", value=1.4)
            pw = st.number_input("عرض گلبرگ (cm):", value=0.2)
            
        if st.button("🚀 محاسبه پیش‌بینی", type="primary"):
            user_input = [[sl, sw, pl, pw]]
            pred = model.predict(user_input)
            st.success(f"🎯 نوع گل: **{iris.target_names[pred[0]]}**")

    # ------------------ سرطان سینه ------------------
    elif st.session_state.dataset == 'cancer':
        st.write("تشخیص خوش‌خیم یا بدخیم بودن توده سرطانی")
        cancer = load_breast_cancer()
        X = cancer.data
        y = cancer.target
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        st.write("برای سادگی، فقط ۳ ویژگی اصلی را وارد کنید:")
        col1, col2, col3 = st.columns(3)
        with col1:
            f1 = st.number_input("شعاع میانگین:", value=14.0)
        with col2:
            f2 = st.number_input("بافت میانگین:", value=19.0)
        with col3:
            f3 = st.number_input("محیط میانگین:", value=90.0)
            
        if st.button("🚀 محاسبه پیش‌بینی", type="primary"):
            user_input = np.zeros((1, 30))
            user_input[0, 0] = f1
            user_input[0, 1] = f2
            user_input[0, 2] = f3
            pred = model.predict(user_input)
            result = "بدخیم (Malignant)" if pred[0] == 0 else "خوش‌خیم (Benign)"
            st.success(f"🎯 نتیجه: **{result}**")

    # ------------------ شراب ------------------
    elif st.session_state.dataset == 'wine':
        st.write("تشخیص نوع شراب بر اساس ویژگی‌های شیمیایی")
        wine = load_wine()
        X = wine.data
        y = wine.target
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            alc = st.number_input("الکل:", value=13.0)
        with col2:
            mal = st.number_input("اسید مالیک:", value=2.0)
        with col3:
            ash = st.number_input("خاکستر:", value=2.5)
            
        if st.button("🚀 محاسبه پیش‌بینی", type="primary"):
            user_input = np.zeros((1, 13))
            user_input[0, 0] = alc
            user_input[0, 1] = mal
            user_input[0, 2] = ash
            pred = model.predict(user_input)
            st.success(f"🎯 نوع شراب: **{wine.target_names[pred[0]]}**")

    if st.button("⬅️ بازگشت به لیست دیتاست‌ها"):
        st.session_state.page = 'menu_classification'
        st.rerun()

# ==========================================
# ۷. صفحه پیش‌بینی کلاستریشن (Iris, Wine)
# ==========================================
def show_predict_clustering():
    st.title("🔮 گروه‌بندی با مدل کلاستریشن (K-Means)")
    
    if st.session_state.dataset == 'iris_cluster':
        st.write("گروه‌بندی گل‌های Iris به ۳ دسته (بدون استفاده از برچسب‌ها)")
        iris = load_iris()
        X = iris.data
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        model.fit(X)
        
        col1, col2 = st.columns(2)
        with col1:
            sl = st.number_input("طول کاسبرگ (cm):", value=5.1)
            sw = st.number_input("عرض کاسبرگ (cm):", value=3.5)
        with col2:
            pl = st.number_input("طول گلبرگ (cm):", value=1.4)
            pw = st.number_input("عرض گلبرگ (cm):", value=0.2)
            
        if st.button("🚀 محاسبه گروه", type="primary"):
            user_input = [[sl, sw, pl, pw]]
            cluster = model.predict(user_input)
            st.success(f"🎯 این گل در گروه (Cluster) شماره **{cluster[0]}** قرار می‌گیرد.")

    elif st.session_state.dataset == 'wine_cluster':
        st.write("گروه‌بندی شراب‌ها به ۳ دسته")
        wine = load_wine()
        X = wine.data
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        model.fit(X)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            alc = st.number_input("الکل:", value=13.0)
        with col2:
            mal = st.number_input("اسید مالیک:", value=2.0)
        with col3:
            ash = st.number_input("خاکستر:", value=2.5)
            
        if st.button("🚀 محاسبه گروه", type="primary"):
            user_input = np.zeros((1, 13))
            user_input[0, 0] = alc
            user_input[0, 1] = mal
            user_input[0, 2] = ash
            cluster = model.predict(user_input)
            st.success(f"🎯 این شراب در گروه (Cluster) شماره **{cluster[0]}** قرار می‌گیرد.")

    if st.button("⬅️ بازگشت به لیست دیتاست‌ها"):
        st.session_state.page = 'menu_clustering'
        st.rerun()

# ==========================================
# ۸. اجرای برنامه (Router)
# ==========================================
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'menu_regression':
    show_menu_regression()
elif st.session_state.page == 'menu_classification':
    show_menu_classification()
elif st.session_state.page == 'menu_clustering':
    show_menu_clustering()
elif st.session_state.page == 'predict_regression':
    show_predict_regression()
elif st.session_state.page == 'predict_classification':
    show_predict_classification()
elif st.session_state.page == 'predict_clustering':
    show_predict_clustering()