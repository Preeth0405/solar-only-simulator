import pandas as pd
import streamlit as st
import numpy as np
import numpy_financial as npf
import plotly.express as px
import json
import plotly.graph_objects as go
from fpdf import FPDF
import plotly.io as pio
import tempfile
import sqlite3
import io
from datetime import datetime

st.set_page_config(layout="wide")

with st.sidebar:
    st.header("🔒 Secure Login")
    st.image("logo.png",width = 150)

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Login logic
    if not st.session_state.authenticated:
        password = st.text_input("Enter password", type="password")
        if password == "pushpower123":
            st.session_state.authenticated = True
            st.rerun()
    else:
        st.success("➜]  Logged in")
        if st.button("⏻ Logout"):
            st.session_state.authenticated = False
            st.rerun()

# --- Access control ---
if not st.session_state.authenticated:
    st.warning("Please enter the correct password to access the app.")
    st.stop()


st.title("ðŸ”† Solar System Simulation + 25-Year Financial Model")

# --- Upload/Download Input Parameters ---
st.sidebar.title("ðŸ’¾ Save or Load Inputs")

uploaded_params = st.sidebar.file_uploader("ðŸ“¤ Upload Parameters (.json)", type="json")
if uploaded_params:
    uploaded_config = json.load(uploaded_params)
    for k, v in uploaded_config.items():
        st.session_state[k] = v
    st.sidebar.success("Inputs loaded from file!")

# --- Setup SQLite database ---
conn = sqlite3.connect("solar_projects.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS projects (
    project_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    csv_load BLOB,
    csv_pv BLOB,
    input_params TEXT
)
""")
conn.commit()

# --- Utility: Get list of projects ---
def get_project_dropdown():
    cursor.execute("SELECT project_name FROM projects")
    projects_list = [row[0] for row in cursor.fetchall()]
    return st.sidebar.selectbox("Select Project", [""] + projects_list)



# --- Upload Section ---
st.header("1. Upload Load and PV Data")
col1, col2 = st.columns(2)
with col1:
    load_file = st.file_uploader("Upload Load Profile (CSV)", type="csv",
                             key="load_file", help="Upload Load data file in vertical format...")
with col2:
    pv_file = st.file_uploader("Upload PV Output Data (CSV)", type="csv",
                           key="pv_file", help="Upload PV output file in vertical format...")

# Inject uploaded buffer if project is loaded
if "load_file_buffer" in st.session_state and load_file is None:
    load_file = st.session_state["load_file_buffer"]
if "pv_file_buffer" in st.session_state and pv_file is None:
    pv_file = st.session_state["pv_file_buffer"]


# --- System Parameters ---
st.header("2. System Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    dc_size = st.number_input("DC System Size (kW)", value=st.session_state.get("dc_size", 40.0), help ="Enter the DC System size in kW, This is used in simulation")
    base_dc_size = st.number_input("Base DC Size in PV File (kW)", value=st.session_state.get("base_dc_size", 40.0),help = "Enter the based DC Size (kW), DC System designed in simulation/ Design platform")
with col2:
    inverter_size = st.number_input("Inverter Capacity (kW)", value=st.session_state.get("inverter_size", 30.0),help = "Enter the Inverter Capacity")
    inverter_eff = st.number_input("Inverter Efficiency (%)", value=st.session_state.get("inverter_eff", 98.0), help = "Enter the Inverter Efficiency (%), If AC output is taken from simulation software, then use efficiency as 100%")/100
with col3:
    export_limit = st.number_input("Export Limit (kW)", value=st.session_state.get("export_limit", 30.0), help = "Export limit as per grid application or assumption")

# --- Utility Rates ---
st.header("3. Utility Tariff Inputs")
col1, col2 = st.columns(2)
with col1:
    import_rate = st.number_input("Import rate (Â£/kWh)", min_value=0.1, value=st.session_state.get("import_rate", 0.25),step=0.01, help = "Import rate of the electricity")
with col2:
    export_rate = st.number_input("Export rate (Â£/kWh)", min_value=0.01,value=st.session_state.get("export_rate", 0.05), step=0.005, help = "Export rate of the electricity")

# --- Financial Parameters ---
st.header("4. Financial Assumptions")
col1, col2 = st.columns(2)
with col1:
    capex_per_kw = st.number_input("Capex (Cost per kW)", value=st.session_state.get("capex_per_kw", 650.0), help = " Price per kW")
    o_and_m_rate = st.number_input("O&M Cost (% of Capex per year)",
                                   value=st.session_state.get("o_and_m_rate", 1.0))/ 100
    apply_degradation = st.checkbox("Apply Degradation", value=st.session_state.get("apply_degradation", False))

    degradation_rate = st.number_input("Degradation per Year (%)",
                                       value=st.session_state.get("degradation_rate", 0.4)) / 100
with col2:
    import_esc = st.number_input("Import Tariff Escalation (%/year)",
                                 value=st.session_state.get("import_esc", 1.0)) / 100
    export_esc = st.number_input("Export Tariff Escalation (%/year)",
                                 value=st.session_state.get("export_esc", 1.0)) / 100
    inflation = st.number_input("General Inflation Rate (%/year)", value=st.session_state.get("inflation", 1.0)) / 100
    esc_year = st.number_input("Electricity Inflation from year ", value=st.session_state.get("esc_year", 8.0))

# --- Save Current Input Parameters ---
if st.sidebar.button("ðŸ“¥ Save Inputs", help = " To save the Inputs to the local server"):
    input_params = {
        "dc_size": dc_size,
        "base_dc_size": base_dc_size,
        "inverter_size":inverter_size,
        "inverter_eff": inverter_eff*100,
        "export_limit": export_limit,
        "import_rate": import_rate,
        "export_rate": export_rate,
        "capex_per_kw": capex_per_kw,
        "o_and_m_rate": o_and_m_rate*100,
        "apply_degradation": apply_degradation,
        "degradation_rate": degradation_rate*100,
        "import_esc": import_esc*100,
        "export_esc": export_esc*100,
        "inflation": inflation*100,
        "esc_year":esc_year
    }

    json_string = json.dumps(input_params, indent=2)
    st.sidebar.download_button("â¬‡ï¸ Download Inputs", json_string, file_name="saved_inputs.json", mime="application/json",help= "Download the Inputs to the local server")

# --- Save Project to Database ---
st.sidebar.title("ðŸ—‚ Project Management")

# Text input for project name
project_name = st.sidebar.text_input("Project Name")

# Save Project button
if st.sidebar.button("ðŸ’¾ Save Project"):
    if not project_name:
        st.sidebar.error("Please enter a project name.")
    elif not load_file or not pv_file:
        st.sidebar.error("Please upload both Load and PV files before saving.")
    else:
        # Read CSV files as bytes
        load_bytes = load_file.getvalue()
        pv_bytes = pv_file.getvalue()

        # Prepare input params dict
        input_params = {
            "dc_size": dc_size,
            "base_dc_size": base_dc_size,
            "inverter_size": inverter_size,
            "inverter_eff": inverter_eff*100,
            "export_limit": export_limit,
            "import_rate": import_rate,
            "export_rate": export_rate,
            "capex_per_kw": capex_per_kw,
            "o_and_m_rate": o_and_m_rate*100,
            "apply_degradation": apply_degradation,
            "degradation_rate": degradation_rate*100,
            "import_esc": import_esc*100,
            "export_esc": export_esc*100,
            "inflation": inflation*100,
            "esc_year": esc_year
        }

        input_params_json = json.dumps(input_params)

        # Insert or update project
        cursor.execute("""
        INSERT OR REPLACE INTO projects (project_name, csv_load, csv_pv, input_params)
        VALUES (?, ?, ?, ?)
        """, (project_name, load_bytes, pv_bytes, input_params_json))
        conn.commit()

        st.sidebar.success(f"Project '{project_name}' saved successfully!")

# --- Load Project from Database ---
st.sidebar.markdown("---")
st.sidebar.subheader("ðŸ“‚ Load Existing Project")

# Get list of project names
selected_project = get_project_dropdown()

if st.sidebar.button("ðŸ“¥ Load Project") and selected_project:
    # Fetch project data
    cursor.execute("""
    SELECT csv_load, csv_pv, input_params
    FROM projects WHERE project_name = ?
    """, (selected_project,))
    row = cursor.fetchone()

    if row:
        load_bytes, pv_bytes, input_params_json = row

        # Load CSVs into dataframes and set as session state
        load_df = pd.read_csv(io.BytesIO(load_bytes))
        pv_df = pd.read_csv(io.BytesIO(pv_bytes))

        # Set file uploader buffers
        st.session_state["load_file_buffer"] = io.BytesIO(load_bytes)
        st.session_state["pv_file_buffer"] = io.BytesIO(pv_bytes)

        # Load input params into session state
        input_params = json.loads(input_params_json)
        for k, v in input_params.items():
            st.session_state[k] = v

        st.session_state["project_loaded"] = True  # optional flag
        st.rerun()  # Force app to re-run with loaded project

        # Optional: Show preview of loaded CSVs
        with st.expander("Preview Loaded Load CSV"):
            st.dataframe(load_df)
        with st.expander("Preview Loaded PV CSV"):
            st.dataframe(pv_df)

# --- Delete Project ---
if st.sidebar.button("ðŸ—‘ï¸ Delete Project") and selected_project:
    confirm = st.sidebar.checkbox(f"Confirm delete '{selected_project}'")

    if confirm:
        cursor.execute("DELETE FROM projects WHERE project_name = ?", (selected_project,))
        conn.commit()
        st.sidebar.success(f"Project '{selected_project}' deleted.")
        st.rerun()  # Force dropdown to refresh


# --- Run Simulation ---
if load_file and pv_file:
    # Check if load_file is BytesIO â†’ handle both cases
    if isinstance(load_file, io.BytesIO):
        load_file.seek(0)
        load_df = pd.read_csv(load_file)
    else:
        load_df = pd.read_csv(load_file)

    if isinstance(pv_file, io.BytesIO):
        pv_file.seek(0)
        pv_df = pd.read_csv(pv_file)
    else:
        pv_df = pd.read_csv(pv_file)

    # Continue your simulation as normal
    df = pd.DataFrame()
    df['Time'] = pd.to_datetime(load_df.iloc[:, 0], dayfirst=True)
    df['Load'] = load_df.iloc[:, 1]
    df['PV_base'] = pv_df.iloc[:, 1]
    df["Month"] = df["Time"].dt.to_period("M")
    df["Hour"] = df["Time"].dt.strftime("%H:%M")

    scaling_factor = dc_size / base_dc_size
    df['PV_Prod'] = df['PV_base'] * scaling_factor
    df['Inv_Limit'] = inverter_size
    df['Clipped'] = (df['PV_Prod'] - df['Inv_Limit']).clip(lower=0)
    df['E_Inv'] = df[['PV_Prod', 'Inv_Limit']].min(axis=1)
    df['E_Use'] = df['E_Inv'] * inverter_eff
    df['Inv_Loss'] = df['E_Inv'] * (1 - inverter_eff)
    df['PV_to_Load'] = df[['E_Use', 'Load']].min(axis=1)
    df['Import'] = (df['Load'] - df['PV_to_Load']).clip(lower=0)
    df['Export'] = (df['E_Use'] - df['PV_to_Load']).clip(lower=0).clip(upper=export_limit)
    df['Excess'] = (df['E_Use'] - df['PV_to_Load'] - df['Export']).clip(lower=0)

    total_pv = df['PV_Prod'].sum()
    total_import = df['Import'].sum()
    total_export = df['Export'].sum()
    total_load = df['Load'].sum()
    base_self_use = df['PV_to_Load'].sum()
    base_export = df['Export'].sum()
    base_self_use_ratio = base_self_use / total_pv if total_pv > 0 else 0
    base_export_ratio = base_export / total_pv if total_pv > 0 else 0
    total_clipped = df['Clipped'].sum()
    total_excess = df['Excess'].sum()
    total_inv_loss = df['Inv_Loss'].sum()
    pv_after_losses = total_pv - total_clipped - total_excess - total_inv_loss
    total_solar_clip = total_clipped + total_excess + total_inv_loss
    loss_in_energy = total_solar_clip / total_pv
    direct_consumption = (base_self_use/pv_after_losses)
    specific_production = total_pv/dc_size
    avg_profile = df.groupby("Hour")["Load"].mean().reset_index(name="Average Load")
    peak_profile = df.groupby("Hour")["Load"].max().reset_index(name="Peak Load")
    avg_production = df.groupby("Hour")["PV_Prod"].mean().reset_index(name="Average PV")
    avg_pv_to_load = df.groupby("Hour")["PV_to_Load"].mean().reset_index(name="Average PV to Load")
    avg_import = df.groupby("Hour")["Import"].mean().reset_index(name="Average Import")
    avg_export  = df.groupby("Hour")["Export"].mean().reset_index(name="Average Export")
    used_on_site = base_self_use / total_load
    imported_from_grid = total_import/total_load


    with st.expander("â˜€ï¸Solar Simulation ResultsðŸ“Š", expanded=True):
        row1 = st.columns(4)
        row1[0].metric("Total PV Production (kWh)", f"{total_pv:.2f}")
        row1[1].metric("Grid Import (kWh)", f"{total_import:.2f}")
        row1[2].metric("Exported Energy (kWh)", f"{total_export:.2f}")
        row1[3].metric("Total Load (kWh)", f"{total_load:.2f}")

        row2=st.columns(4)
        row2[0].metric("PV used on site (kWh)",f"{base_self_use:.2f}")
        row2[1].metric("Cipped Energy (kWh)",f"{total_clipped:.2f}")
        row2[2].metric("Excess Energy (kWh)",f"{total_excess:.2f}")
        row2[3].metric("Inverter Losses (kWh)",f"{total_inv_loss:.2f}")

        row3=st.columns(4)
        row3[0].metric("PV used on site (%)",f"{(base_self_use/total_load)*100:.2f}%")
        row3[1].metric("Exported Energy (%)",f"{base_export_ratio * 100:.2f}%")
        row3[2].metric("Imported Energy (%)",f"{(total_import/total_load)*100:.2f}%")
        row3[3].metric("Self Consumption (%)",f"{base_self_use_ratio*100:.2f}%")

        row4=st.columns(4)
        row4[0].metric("Excess Energy (%)",f"{(total_excess/total_pv)*100:.2f}%")
        row4[1].metric("Clipped Energy (%)",f"{(total_clipped/total_pv) * 100:.2f}%")
        row4[2].metric("Inverter Losses (%)",f"{(total_inv_loss/total_pv)*100:.2f}%")
        row4[3].metric("Yield (kWh/kWp)",f"{total_pv/dc_size:.2f}")


        df['Date'] = df['Time'].dt.date
        df['Hour'] = df['Time'].dt.hour

    daily_summary = df.groupby('Time').agg({
        'Load': 'sum',
        'PV_Prod': 'sum',
        'PV_to_Load': 'sum',
        'Import': 'sum',
        'Export': 'sum',
        'Excess': 'sum',
        'Clipped': 'sum',
        'Inv_Loss': 'sum'
    }).reset_index()

    st.sidebar.subheader("Customize Chart Colors")
    load_color = st.sidebar.color_picker("Load","#EF553B")
    pv_prod_color = st.sidebar.color_picker("PV Production", "#636EFA")
    pv_to_load_color = st.sidebar.color_picker("PV to Load", "#00CC96")

    with st.expander("ðŸ“ŠCharts & GraphsðŸ“‰", expanded=False):

        fig1 = px.line(daily_summary, x='Time', y=['Load', 'PV_Prod', 'PV_to_Load'], title="Daily Load vs PV",color_discrete_map={
        'Load': load_color,
        'PV_Prod': pv_prod_color,
        'PV_to_Load': pv_to_load_color})
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(daily_summary, x='Time', y=['Import', 'Export', 'Excess'], title="Import, Export, Excess")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(daily_summary, x='Time', y=['PV_Prod', 'Clipped', 'Inv_Loss'], title="Production Losses")
        st.plotly_chart(fig3, use_container_width=True)


        selected_day = st.date_input("Select day", value=min(df['Date']),min_value=min(df['Date']),max_value=max(df['Date']))

        daily_filtered = df[df['Date'] == selected_day]

        fig4 = px.line(daily_filtered,x='Time',y=['Load','PV_Prod','PV_to_Load','Import','Export','Excess'], title=f"Energy profile for {selected_day}", labels={'value':'Energy (kWh)', 'Time':'Time of Day'})
        st.plotly_chart(fig4, use_container_width=True)

    sim_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Solar Simulation CSV", sim_csv, "solar_simulation.csv", "text/csv")

    # --- Financial Projection ---
    st.header("5. 25-Year Financial Results")
    initial_capex = dc_size * capex_per_kw
    years = list(range(26))
    degradation_factors = [(1 - degradation_rate) ** (y - 1) if apply_degradation and y > 0 else 1.0 for y in years]

    cashflow = []
    cumulative = -initial_capex

    for y in years:
        if y == 0:
            cashflow.append({
                "Year": 0,
                "System Price (Â£)": -initial_capex,
                "O&M Costs (Â£)": 0,
                "Net Bill Savings (Â£)": 0,
                "Export Income (Â£)": 0,
                "Annual Cash Flow (Â£)": -initial_capex,
                "Cumulative Cash Flow (Â£)": -initial_capex,
                "PV Production":pv_after_losses,
                "Export Energy":total_export,
                "Import rates":import_rate,
                "Export rates":export_rate
            })
            continue

        deg = degradation_factors[y]
        pv_prod = (dc_size * specific_production - total_solar_clip) * deg
        pv_to_load = pv_prod * direct_consumption
        pv_export = pv_prod - pv_to_load
        import_required = total_load - pv_to_load

        imp_price = import_rate * ((1 + import_esc) ** max(0, y - esc_year))
        exp_price = export_rate * ((1 + export_esc) ** max(0,y - esc_year))

        savings = (total_load - import_required) * imp_price
        export_income = pv_export * exp_price
        om = initial_capex * o_and_m_rate * ((1 + inflation) ** (y - 1))

        annual_cashflow = savings + export_income - om
        cumulative += annual_cashflow

        cashflow.append({
            "Year": y,
            "System Price (Â£)": -initial_capex if y == 0 else 0,
            "O&M Costs (Â£)": -om if y > 0 else 0,
            "Net Bill Savings (Â£)": savings,
            "Export Income (Â£)": export_income,
            "Annual Cash Flow (Â£)": annual_cashflow,
            "Cumulative Cash Flow (Â£)": cumulative,
            "PV Production":pv_prod,
            "Export Energy":pv_export,
            "Import rates":imp_price,
            "Export rates":exp_price
        })

    fin_df = pd.DataFrame(cashflow)
    irr = npf.irr(fin_df['Annual Cash Flow (Â£)'])
    roi = (fin_df['Cumulative Cash Flow (Â£)'].iloc[-1] + initial_capex) / initial_capex

    payback = None
    payback_display = "Not achieved"
    for i in range(1, len(fin_df)):
        if fin_df.loc[i, 'Cumulative Cash Flow (Â£)'] >= 0:
            prev_cum = fin_df.loc[i - 1, 'Cumulative Cash Flow (Â£)']
            annual_cash = fin_df.loc[i, 'Annual Cash Flow (Â£)']
            if annual_cash != 0:
                payback = i - 1 + abs(prev_cum) / annual_cash
                years = int(payback)
                months = int(round((payback - years) * 12))
                payback_display = f"{years} years {months} months"
            break

    lcoe = initial_capex / sum([total_pv * d for d in degradation_factors[1:]])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Initial Capex (Â£)", f"{initial_capex:,.2f}")
    col2.metric("Payback Period", payback_display)
    col3.metric("ROI (%)", f"{roi * 100:.2f}")
    col4.metric("IRR (%)", f"{irr * 100:.2f}")
    col5.metric("LCOE (Â£/kWh)", f"{lcoe:.2f}")

    with st.expander("ðŸ“‹ Show Cash Flow Table"):
        st.dataframe(fin_df.style.format({
            "System Price (Â£)": "Â£{:,.2f}",
            "O&M Costs (Â£)": "Â£{:,.2f}",
            "Net Bill Savings (Â£)": "Â£{:,.2f}",
            "Export Income (Â£)": "Â£{:,.2f}",
            "Annual Cash Flow (Â£)": "Â£{:,.2f}",
            "Cumulative Cash Flow (Â£)": "Â£{:,.2f}"
        }))

    with st.expander("ðŸ’°Financial ChartðŸ“ˆ"):
     st.plotly_chart(px.bar(fin_df[1:], x='Year', y='Annual Cash Flow (Â£)', title="Annual Cash Flow"), use_container_width=True)
     st.plotly_chart(px.line(fin_df[1:], x='Year', y='Cumulative Cash Flow (Â£)', title="Cumulative Cash Flow"), use_container_width=True)

    csv = fin_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cash Flow Table", csv, "cashflow_25yr.csv", "text/csv")

    simulate_batch =st.radio("Batch Simulation",["No","Yes"],index=0,horizontal = True)
    if simulate_batch == "Yes":
        with st.expander("ðŸ“Š Batch Simulation (Compare Multiple Systems)", expanded=False):
            num_systems = st.number_input("How many systems to compare?", min_value=2, max_value=100, value=3, step=1)
            dc_num = st.number_input("DC increment ?", min_value = 10, max_value = 100, value = 10, step = 10)
            ac_num = st.number_input("AC increment ?", min_value = 10, max_value = 100, value = 10, step = 10)

        st.markdown("### System Parameters")
        batch_data = {
            "System": [f"System {i+1}" for i in range(num_systems)],
            "DC Size (kW)": [dc_size + dc_num * i for i in range(num_systems)],
            "AC Size (kW)": [inverter_size + ac_num * i for i in range(num_systems)],
            "Export Limit (kW)": [export_limit for _ in range(num_systems)],
        }

        batch_df = pd.DataFrame(batch_data)

        for i in range(num_systems):
            col1, col2, col3 = st.columns(3)
            with col1:
                batch_df.at[i, "DC Size (kW)"] = st.number_input(f"DC Size - System {i+1}", key=f"dc_{i}", value=batch_df.at[i, "DC Size (kW)"])
            with col2:
                batch_df.at[i, "AC Size (kW)"] = st.number_input(f"AC Size - System {i+1}", key=f"ac_{i}", value=batch_df.at[i, "AC Size (kW)"])
            with col3:
                batch_df.at[i, "Export Limit (kW)"] = st.number_input(f"Export Limit - System {i+1}", key=f"exp_{i}", value=batch_df.at[i, "Export Limit (kW)"])

        batch_df["DC/AC Ratio"] = (batch_df["DC Size (kW)"] / batch_df["AC Size (kW)"]).round(2)

        st.dataframe(batch_df)

        # --- Run Batch Calculations ---
        comparison_results = []
        for i, row in batch_df.iterrows():
            dc = row["DC Size (kW)"]
            ac = row["AC Size (kW)"]
            exp_limit = row["Export Limit (kW)"]
            dc_ac_ratio = row["DC/AC Ratio"]
            scaling = dc / base_dc_size

            temp_df = pd.DataFrame()
            temp_df['Load'] = df['Load']
            temp_df['PV_base'] = df['PV_base']
            temp_df['PV_Prod'] = df['PV_base'] * scaling
            temp_df['Inv_Limit'] = ac
            temp_df['E_Inv'] = temp_df[['PV_Prod', 'Inv_Limit']].min(axis=1)
            temp_df['E_Use'] = temp_df['E_Inv'] * inverter_eff
            temp_df['PV_to_Load'] = temp_df[['E_Use', 'Load']].min(axis=1)
            temp_df['Import'] = (temp_df['Load'] - temp_df['PV_to_Load']).clip(lower=0)
            temp_df['Export'] = (temp_df['E_Use'] - temp_df['PV_to_Load']).clip(lower=0).clip(upper=exp_limit)

            total_pv_batch = temp_df['PV_Prod'].sum()
            pv_self = temp_df['PV_to_Load'].sum()
            pv_export = temp_df['Export'].sum()
            self_ratio = (pv_self / total_pv_batch)*100 if total_pv_batch > 0 else 0
            exp_ratio = (pv_export / total_pv_batch)*100 if total_pv_batch > 0 else 0

            capex = dc * capex_per_kw
            om_cost = capex * o_and_m_rate
            net_annual = pv_self * import_rate + pv_export * export_rate - om_cost
            irr = npf.irr([-capex] + [net_annual] * 25)
            roi = ((net_annual * 25) - capex) / capex
            lcoe = capex / total_pv_batch

            cum_cash = -capex
            payback = None
            for yr in range(1, 26):
                cum_cash += net_annual
                if cum_cash >= 0:
                    payback = yr
                    break

            comparison_results.append({
                "System": f"System {i+1}",
                "Capex (Â£)": f"Â£{capex:,.0f}",
                "DC Size (kW)": f"{dc:,.0f}",
                "AC Size (kW)": f"{ac:,.0f}",
                "Export Limit (kW)": exp_limit,
                "DC/AC Ratio": dc_ac_ratio,
                "Total PV (kWh)": round(total_pv_batch, 1),
                "Self-Use Ratio (%)": round(self_ratio, 2),
                "Export Ratio (%)": round(exp_ratio, 2),
                "Payback (yrs)": payback if payback else "N/A",
                "ROI (%)": round(roi * 100, 1),
                "IRR (%)": round(irr * 100, 1) if irr is not None else "N/A",
            })

        st.subheader("ðŸ“‹ Batch Comparison Table")
        comp_df = pd.DataFrame(comparison_results)
        st.dataframe(comp_df)

        st.subheader("ðŸ“ˆ Compare Metric Across Systems")
        metric_option = st.selectbox("Select Metric to Plot", ["Payback (yrs)", "ROI (%)", "IRR (%)", "LCOE (Â£/kWh)"])
        fig = px.bar(comp_df, x="System", y=metric_option, title=f"{metric_option} Comparison")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("â™»ï¸CarbonðŸ­"):
        carbon_index = 0.20705
        before_project = total_load * carbon_index
        after_project = total_import * carbon_index
        carbon_saved = before_project - after_project
        carbon_units = st.selectbox("Units of Carbon",["kg","Tonne"])
        carbon_unit = 1000 if carbon_units == "Tonne" else 1

        col1, col2,col3,col4,col5,col6,col7 = st.columns(7)
        with col1:
            st.metric(f"Before Project âš¡ï¸({carbon_units})",f"{before_project/carbon_unit:,.2f} ")
            st.image("https://img.icons8.com/ios-filled/100/000000/co2.png",width=60)
        with col2:
            st.metric(f"After Project ðŸŒž({carbon_units})", f"{after_project / carbon_unit:,.2f} ")
            st.image("https://img.icons8.com/ios-filled/100/000000/solar-panel.png", width=60)
        with col3:
            st.metric(f"Carbon Saved â™»ï¸({carbon_units})", f"{carbon_saved / carbon_unit:,.2f} ")
            st.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/external-carbon-footprint-lifestyles-flaticons-lineal-color-flat-icons.png",width=60)
        with col4:
            st.metric(f"Trees Saved ðŸŒ´",f"{carbon_saved/22:,.0f}")
            st.image("https://img.icons8.com/clouds/100/bonsai.png",width=60)
        with col5:
            st.metric("Cars off the Road ðŸš˜",f"{carbon_saved/1900:,.0f}")
            st.image("https://img.icons8.com/plasticine/100/traffic-jam.png",width=60)
        with col6:
            st.metric("Miles Driven ðŸ›£",f"{carbon_saved/0.168:,.0f}")
            st.image("https://img.icons8.com/emoji/48/motorway.png",width=60)
        with col7:
            st.metric("Homes Powered ðŸ¡",f"{carbon_saved/2800:,.0f}")
            st.image("https://img.icons8.com/color/48/mansion.png",width=60)

    with st.expander("ðŸ“ˆAverage ProfileðŸ“‰"):
        avg_df = avg_profile \
        .merge(avg_production, on = "Hour")\
        .merge(avg_pv_to_load, on = "Hour")\
        .merge(avg_import, on = "Hour")\
        .merge(avg_export, on = "Hour")

        st.plotly_chart(px.line(avg_df, x="Hour", y=['Average Load','Average PV','Average PV to Load','Average Import','Average Export'], title="Average Load Over Time"),
                        use_container_width=True)
        st.plotly_chart(px.line(peak_profile, x="Hour", y="Peak Load", title="Peak Load Over Time"),
                        use_container_width=True)

    if "Month" not in df.columns:
        df["Month"] = df["Time"].dt.strftime("%B")

    monthly = df.groupby("Month").agg({"Load": "sum", "PV_Prod": "sum", "PV_to_Load": "sum",
             "Import": "sum", "Export": "sum","Excess": "sum","Clipped" :"sum","Inv_Loss" : "sum"
        }).rename(columns={
            "Load": "Load (kWh)", "PV_Prod": "Production (kWh)", "PV_to_Load": "Solar On-site (kWh)",
            "Import": "Grid (kWh)", "Export": "Export (kWh)","Excess": "Excess (kWh)","Clipped":"Clipped Energy (kWh)","Inv_Loss":"Inverter Losses (kWh)"
        }).reset_index()

    monthly["Month"] = monthly["Month"].astype(str)

    with st.expander("ðŸ“… Monthly Summary Table", expanded=False):
        st.dataframe(monthly)

    with st.expander("ðŸ“… Monthly Summary Chart", expanded=False):

        monthly["Grid Before"] = monthly["Load (kWh)"]
        monthly["Grid After"] = monthly["Grid (kWh)"]
        monthly["Renewable"] = monthly["Solar On-site (kWh)"]
        monthly["Export"] = monthly["Export (kWh)"]
        monthly["Bill Before "] = monthly["Load (kWh)"] * import_rate
        monthly["Bill After"] = monthly["Grid (kWh)"] * import_rate

        # Plot
        fig = go.Figure()

        # Background bar (Total Load)
        fig.add_trace(go.Bar(
            x=monthly["Month"],
            y=monthly["Grid Before"],
            name="Site Load (kWh)",
            marker_color="#F2F2F2",
            opacity=0.5
        ))

        # Grid Purchase Before Solar (dotted)
        fig.add_trace(go.Scatter(
            x=monthly["Month"],
            y=monthly["Grid Before"],
            mode='lines',
            name="Grid Purchase - Before Solar",
            line=dict(dash='dot', color='#543053')
        ))

        # Grid Purchase After Solar (solid line)
        fig.add_trace(go.Scatter(
            x=monthly["Month"],
            y=monthly["Grid After"],
            mode='lines',
            name="Grid Purchase - After Solar",
            line=dict(color='#7F2A63')
        ))

        # Renewable Consumption (solid)
        fig.add_trace(go.Scatter(
            x=monthly["Month"],
            y=monthly["Renewable"],
            mode='lines',
            name="Renewable Consumption",
            line=dict(color='#92AD3D')
        ))

        # Export (dashed)
        fig.add_trace(go.Scatter(
            x=monthly["Month"],
            y=monthly["Export"],
            mode='lines',
            name="Export",
            line=dict(dash='dash', color='#587434')
        ))

        # Layout styling
        fig.update_layout(
            title="Site Consumption (kWh)",
            xaxis_title="Month",
            yaxis_title="Energy (kWh)",
            template="plotly_dark",
            barmode='overlay',
            legend=dict(orientation="h", y=-0.5),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        bill_fig = go.Figure()

        bill_fig.add_trace(go.Bar(
            x=monthly["Month"],
            y=monthly["Bill Before "],
            name="Before Solar",
            marker_color="#858F8F"  # grey
        ))

        # After Solar (green bar)
        bill_fig.add_trace(go.Bar(
            x=monthly["Month"],
            y=monthly["Bill After"],
            name="After Solar",
            marker_color="#92AD3D"  # green
        ))

        # Layout
        bill_fig.update_layout(
            title="Approximate Electricity Bill - Year 1",
            xaxis_title="Month",
            yaxis_title="Cost (Â£)",
            barmode='group',  # for side-by-side bars
            template="plotly_dark",  # or "simple_white" for white background
            height=450,
            yaxis_tickprefix="Â£",
            legend=dict(orientation="h", y=-0.2)
        )

        st.plotly_chart(bill_fig, use_container_width=True)

        fig_donut = go.Figure(data=[go.Pie(labels=["Solar","Grid Import"],values = [used_on_site,imported_from_grid],hole=0.5, marker=dict(colors=["#543053","#858F8F"]),textinfo="percent",insidetextorientation="horizontal")])

        fig_donut.update_layout(title = "Energy Source",template="plotly_dark",showlegend=True,height=450)

        fig_carbon = go.Figure(data=[go.Pie(labels=["Before","After"],values = [(before_project/1000),(after_project/1000)],hole =0.5,marker = dict(colors=["858F85","#543053"]),texttemplate = "<br>%{value:.2f} tCOâ‚‚",textinfo="value",insidetextorientation="horizontal")])
        fig_carbon.update_layout(title = "Carbon Emissions",template="plotly_dark",showlegend=True,height=450)

        col1,col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_donut, use_container_width=True)
        with col2:
            st.plotly_chart(fig_carbon, use_container_width=True)



        # --- AI Optimisation Section ---
        st.header("🔎 AI Optimisation (with Constraints & Metric Choice)")

        import random

        # --- User-defined ranges ---
        st.subheader("Constraint Settings")

        # PV system size
        pv_min = st.number_input("Min PV Size (kWp)", value=0.8*base_dc_size, step=1.0)
        pv_max = st.number_input("Max PV Size (kWp)", value=1.5*base_dc_size, step=1.0)

        # Self-consumption
        sc_min = st.number_input("Min Self-Consumption (%)", value=50.0, step=1.0)
        sc_max = st.number_input("Max Self-Consumption (%)", value=100.0, step=1.0)

        # Export %
        exp_min = st.number_input("Min Export (%)", value=0.0, step=1.0)
        exp_max = st.number_input("Max Export (%)", value=50.0, step=1.0)

        # DC/AC Ratio
        dcr_min = st.number_input("Min DC/AC Ratio", value=1.0, step=0.1)
        dcr_max = st.number_input("Max DC/AC Ratio", value=1.5, step=0.1)

        # Payback
        pb_min = st.number_input("Min Payback (yrs)", value=0.0, step=1.0)
        pb_max = st.number_input("Max Payback (yrs)", value=20.0, step=1.0)

        # Export limit range (kW)
        exp_limit_min = st.number_input("Min Export Limit (kW)", value=5.0, step=1.0)
        exp_limit_max = st.number_input("Max Export Limit (kW)", value=float(inverter_size), step=1.0)

        # Optimisation metric
        optimise_for = st.selectbox(
            "Optimise for:",
            ["IRR (%)", "Payback (yrs)", "Self-Use (%)", "Site Consumption (%)", "Export (%)", "DC/AC Ratio"]
        )

        # --- Simulation Function ---
        def run_simulation_once(dc_size, inverter_size, export_limit):
            inverter_size = round(inverter_size)
            export_limit = round(export_limit)

            scaling_factor = dc_size / base_dc_size
            temp_df = pd.DataFrame()
            temp_df['Load'] = df['Load']
            temp_df['PV_base'] = df['PV_base']
            temp_df['PV_Prod'] = df['PV_base'] * scaling_factor
            temp_df['Inv_Limit'] = inverter_size
            temp_df['E_Inv'] = temp_df[['PV_Prod', 'Inv_Limit']].min(axis=1)
            temp_df['E_Use'] = temp_df['E_Inv'] * inverter_eff
            temp_df['PV_to_Load'] = temp_df[['E_Use', 'Load']].min(axis=1)
            temp_df['Import'] = (temp_df['Load'] - temp_df['PV_to_Load']).clip(lower=0)
            temp_df['Export'] = (temp_df['E_Use'] - temp_df['PV_to_Load']).clip(lower=0).clip(upper=export_limit)

            pv_self = temp_df['PV_to_Load'].sum()
            pv_export = temp_df['Export'].sum()
            total_pv = temp_df['PV_Prod'].sum()
            total_load = temp_df['Load'].sum()

            capex = dc_size * capex_per_kw
            om_cost = capex * o_and_m_rate
            net_annual = pv_self * import_rate + pv_export * export_rate - om_cost
            irr = npf.irr([-capex] + [net_annual] * 25)

            # Payback
            cum_cash = -capex
            payback = None
            for yr in range(1, 26):
                cum_cash += net_annual
                if cum_cash >= 0:
                    payback = yr
                    break

            # Ratios
            self_use_pct = (pv_self / total_pv * 100) if total_pv > 0 else 0
            site_consumption_pct = (pv_self / total_load * 100) if total_load > 0 else 0
            export_pct = (pv_export / total_pv * 100) if total_pv > 0 else 0
            dc_ac_ratio = dc_size / inverter_size if inverter_size > 0 else None

            return {
                "DC Size (kW)": round(dc_size, 1),
                "Inverter Size (kW)": inverter_size,
                "Export Limit (kW)": export_limit,
                "IRR (%)": irr * 100 if irr else None,
                "Payback (yrs)": payback,
                "Self-Use (%)": self_use_pct,
                "Site Consumption (%)": site_consumption_pct,
                "Export (%)": export_pct,
                "DC/AC Ratio": dc_ac_ratio
            }

        # --- Optimiser Loop ---
        def optimise_system(trials=200):
            best = None
            history = []
            for _ in range(trials):
                dc = random.uniform(pv_min, pv_max)
                inv = random.uniform(0.6*dc, 1.0*dc)
                exp = random.uniform(exp_limit_min, min(exp_limit_max, inv, dc))  # fix export ≤ inverter ≤ DC

                result = run_simulation_once(dc, inv, exp)
                history.append(result)

                # Apply constraints
                valid = True
                valid &= pv_min <= result["DC Size (kW)"] <= pv_max
                valid &= sc_min <= result["Self-Use (%)"] <= sc_max
                valid &= exp_min <= result["Export (%)"] <= exp_max
                valid &= dcr_min <= result["DC/AC Ratio"] <= dcr_max
                if result["Payback (yrs)"] is not None:
                    valid &= pb_min <= result["Payback (yrs)"] <= pb_max

                if not valid:
                    continue

                # --- Optimisation metric selection ---
                if best is None:
                    best = result
                else:
                    if optimise_for in ["IRR (%)", "Self-Use (%)", "Site Consumption (%)"]:
                        if result[optimise_for] > best[optimise_for]:
                            best = result
                    elif optimise_for in ["Export (%)", "Payback (yrs)"]:
                        if result[optimise_for] < best[optimise_for]:
                            best = result
                    elif optimise_for == "DC/AC Ratio":
                        if abs(result["DC/AC Ratio"] - round(result["DC/AC Ratio"])) < abs(best["DC/AC Ratio"] - round(best["DC/AC Ratio"])):
                            best = result

            return best, pd.DataFrame(history)

        # --- Run Button ---
        if st.button("🚀 Run Optimiser"):
            best_sol, hist_df = optimise_system(trials=200)
            if best_sol:
                st.success(f"Best System Found ({optimise_for} within constraints) → "
                           f"DC {best_sol['DC Size (kW)']} kW, "
                           f"Inverter {best_sol['Inverter Size (kW)']} kW, "
                           f"Export {best_sol['Export Limit (kW)']} kW, "
                           f"{optimise_for} = {best_sol[optimise_for]:.2f}")
            else:
                st.error("No system found that meets your constraints.")

            st.subheader("Optimisation History")
            st.dataframe(hist_df)
            if optimise_for in hist_df.columns:
                st.line_chart(hist_df[optimise_for])






        import psutil
        import os
        import streamlit as st


        def get_memory_usage():
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_used_mb = mem_info.rss / (1024 ** 2)  # in MB
            return mem_used_mb


        st.sidebar.markdown(f"**Memory Usage:** {get_memory_usage():.2f} MB")



else:
    st.warning("Please upload both Load and PV files to run the simulation.")
