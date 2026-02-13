import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os
import platform
from pathlib import Path

# Add current directory to path to import simulate_auction
sys.path.append(os.path.dirname(__file__))
import simulate_auction

# Configure Matplotlib to use Chinese fonts
system_name = platform.system()
if system_name == "Linux":
    # Streamlit Cloud (Linux) uses fonts-wqy-microhei
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
elif system_name == "Darwin": # MacOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False
else: # Windows
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

# Page Configuration
st.set_page_config(
    page_title="数字经济仿真实验：二级密封拍卖",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Sidebar
st.sidebar.title("实验导航")
page = st.sidebar.radio("选择页面", ["实验大纲", "仿真实验", "结果分析"])

# Helper function to load syllabus
def load_syllabus():
    with open("syllabus.md", "r", encoding="utf-8") as f:
        return f.read()

# 1. Syllabus Page
if page == "实验大纲":
    st.title("📚 实验大纲：二级密封拍卖仿真实验设计")
    st.markdown(load_syllabus())
    
    with open("实验大纲：二级密封拍卖仿真实验设计.docx", "rb") as f:
        st.download_button(
            label="📥 下载实验大纲完整文档 (DOCX)",
            data=f,
            file_name="实验大纲：二级密封拍卖仿真实验设计.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

# 2. Simulation Page
elif page == "仿真实验":
    st.title("🖥️ 仿真实验运行")
    st.markdown("在这里设置参数并运行二级密封拍卖的仿真过程。")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("参数设置")
        seed = st.number_input("随机种子 (Seed)", value=42, step=1)
        auctions = st.number_input("拍卖场次 (Auctions)", value=100, step=10)
        bidders = st.number_input("投标者人数 (Bidders)", value=4, step=1)
        
        scenarios_input = st.text_input("场景设置 (ID:Sigma)", value="1:10, 2:30", help="格式: ID:Sigma, 用逗号分隔")
        
        quality_low = st.number_input("质量下限", value=0.0)
        quality_high = st.number_input("质量上限", value=100.0)
        reserve_price = st.number_input("保留价格", value=0.0)
        
        run_btn = st.button("🚀 开始仿真", type="primary")

    with col2:
        if run_btn:
            with st.spinner("正在运行仿真..."):
                try:
                    # Parse scenarios
                    scenarios = simulate_auction.parse_scenarios(scenarios_input)
                    
                    # Generate data
                    rows = simulate_auction.generate_rows(
                        seed=seed,
                        auctions=auctions,
                        bidders=bidders,
                        scenarios=scenarios,
                        quality_low=quality_low,
                        quality_high=quality_high,
                        reserve_price=reserve_price
                    )
                    
                    # Convert to DataFrame for easier handling
                    df = pd.DataFrame(rows)
                    st.session_state['data'] = df
                    st.session_state['scenarios'] = scenarios
                    st.session_state['params'] = {
                        "seed": seed, "auctions": auctions, "bidders": bidders,
                        "quality_low": quality_low, "quality_high": quality_high,
                        "reserve_price": reserve_price
                    }
                    
                    # Save files (mimic main logic)
                    simulate_auction.write_csv(Path("data_exp4_auction.csv"), rows, simulate_auction.FIELDNAMES)
                    
                    # Run Analysis (OLS & Overbid)
                    reg_rows = simulate_auction.ols_with_hc1(rows)
                    overbid_rows = simulate_auction.summarize_overbid(rows)
                    
                    st.session_state['reg_rows'] = reg_rows
                    st.session_state['overbid_rows'] = overbid_rows
                    
                    # Save other files
                    # (Simplified for web app, we generate on the fly or just use the data)
                    
                    st.success("仿真完成！数据已生成。")
                    
                    # Display Execution Summary
                    st.subheader("执行摘要")
                    st.write(f"**总行数**: {len(rows)}")
                    st.write(f"**场景**: {scenarios_input}")
                    
                    st.subheader("数据预览 (前10行)")
                    st.dataframe(df.head(10))
                    
                    # Download Data
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 下载仿真数据 (CSV)",
                        csv,
                        "data_exp4_auction.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                except Exception as e:
                    st.error(f"仿真出错: {e}")
        
        # If data exists in session, show it
        elif 'data' in st.session_state:
            st.info("已加载上次仿真的数据。")
            st.dataframe(st.session_state['data'].head(10))

# 3. Analysis Page
elif page == "结果分析":
    st.title("📈 实验结果分析")
    
    if 'data' not in st.session_state:
        st.warning("请先在“仿真实验”页面运行仿真以生成数据。")
    else:
        df = st.session_state['data']
        reg_rows = st.session_state.get('reg_rows', [])
        overbid_rows = st.session_state.get('overbid_rows', [])
        
        tab1, tab2, tab3 = st.tabs(["回归分析", "赢家诅咒 (Overbid)", "流程图"])
        
        with tab1:
            st.header("回归分析: 成交价 vs 质量")
            st.markdown("模型: $Price = \\alpha + \\beta \\cdot Quality + \\epsilon$")
            
            # Display Regression Table
            if reg_rows:
                reg_df = pd.DataFrame(reg_rows)
                # Format columns
                reg_df = reg_df[['term', 'coef', 'robust_se', 't_value', 'p_value']]
                st.table(reg_df.style.format({
                    'coef': '{:.4f}', 'robust_se': '{:.4f}', 
                    't_value': '{:.4f}', 'p_value': '{:.4f}'
                }))
            
            # Scatter Plot
            st.subheader("散点图与拟合线")
            # Using simple matplotlib for regression line
            fig, ax = plt.subplots()
            ax.scatter(df['Quality'], df['Price'], alpha=0.5, label='Data')
            
            # Plot regression line if coefficients exist
            if reg_rows:
                beta0 = next((r['coef'] for r in reg_rows if r['term'] == '_cons'), 0)
                beta1 = next((r['coef'] for r in reg_rows if r['term'] == 'Quality'), 0)
                x_vals = [df['Quality'].min(), df['Quality'].max()]
                y_vals = [beta0 + beta1 * x for x in x_vals]
                ax.plot(x_vals, y_vals, 'r-', label=f'Fit: y={beta0:.2f}+{beta1:.2f}x')
            
            ax.set_xlabel("Quality")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
            
        with tab2:
            st.header("赢家诅咒 (Winner's Curse) 分析")
            st.markdown("分析不同场景下获胜者出价高于真实价值 (Overbid) 的情况。")
            
            # Overbid Summary Table
            if overbid_rows:
                st.subheader("各场景统计汇总")
                overbid_df = pd.DataFrame(overbid_rows)
                st.table(overbid_df.style.format({
                    'mean_overbid': '{:.4f}', 'std_overbid': '{:.4f}',
                    'median_overbid': '{:.4f}', 'max_overbid': '{:.4f}'
                }))
            
            # Box Plot
            st.subheader("Overbid 分布 (箱线图)")
            # Filter only winners for meaningful overbid analysis if needed, 
            # but the script calculates overbid for winners and 0 for others.
            # Usually we analyze overbid for winners or all? 
            # The script says: "Overbid definition for winners only... cond(Win==1, ...)"
            # Let's use the 'Overbid' column from dataframe directly as it matches script logic.
            
            fig_box, ax_box = plt.subplots()
            df.boxplot(column='Overbid', by='scenario', ax=ax_box)
            ax_box.set_ylabel("Overbid Amount")
            ax_box.set_title("Overbid by Scenario")
            plt.suptitle("") # Remove default pandas suptitle
            st.pyplot(fig_box)
            
            # Bar Plot
            st.subheader("平均 Overbid (柱状图)")
            if overbid_rows:
                fig_bar, ax_bar = plt.subplots()
                scenarios = [str(int(r['scenario'])) for r in overbid_rows]
                means = [r['mean_overbid'] for r in overbid_rows]
                ax_bar.bar(scenarios, means, color=['skyblue', 'salmon', 'lightgreen'])
                ax_bar.set_xlabel("Scenario")
                ax_bar.set_ylabel("Mean Overbid")
                st.pyplot(fig_bar)

        with tab3:
            st.header("仿真流程图")
            # The script generates an SVG. We can generate it or load it.
            # Let's generate it using the function from script
            svg_path = Path("auction_flowchart.svg")
            simulate_auction.write_flowchart_svg(svg_path)
            
            if svg_path.exists():
                st.image(str(svg_path), caption="Auction Simulation Flowchart", use_container_width=True)
            else:
                st.error("流程图生成失败")

