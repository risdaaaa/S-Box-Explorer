import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from io import BytesIO

# Function konversi output S-box ke bentuk biner 
def sbox_to_binary(sbox, m):
    return np.array([[int(bit) for bit in format(val, f'0{m}b')] for val in sbox])

# Walsh-Hadamard Transform (WHT) Function
def walsh_hadamard_transform(func):
    size = len(func)
    wht = np.copy(func) * 2 - 1
    step = 1
    while step < size:
        for i in range(0, size, step * 2):
            for j in range(step):
                u = wht[i + j]
                v = wht[i + j + step]
                wht[i + j] = u + v
                wht[i + j + step] = u - v
        step *= 2
    return wht

# Nonlinearity (NL) Function
def compute_nl(binary_sbox, n, m):
    nl_values = []
    for i in range(m):
        func = binary_sbox[:, i]
        wht = walsh_hadamard_transform(func)
        max_correlation = np.max(np.abs(wht))
        nl = 2**(n-1) - (max_correlation // 2)
        nl_values.append(nl)
    return min(nl_values), nl_values

# Strict Avalanche Criterion (SAC) Function
def compute_sac(binary_sbox, n, m):
    num_inputs = 2**n
    sac_matrix = np.zeros((n, m))
    for i in range(num_inputs):
        original_output = binary_sbox[i]
        for bit in range(n):
            flipped_input = i ^ (1 << bit)
            flipped_output = binary_sbox[flipped_input]
            bit_changes = original_output ^ flipped_output
            sac_matrix[bit] += bit_changes
    sac_matrix /= num_inputs
    return np.mean(sac_matrix), sac_matrix

# Bit Independence Criterion—Nonlinearity (BIC-NL) Function
def compute_bic_nl(binary_sbox, n, m):
    bic_nl_values = []
    for i in range(m):
        for j in range(i + 1, m):
            combined_func = binary_sbox[:, i] ^ binary_sbox[:, j]
            wht = walsh_hadamard_transform(combined_func)
            max_correlation = np.max(np.abs(wht))
            nl = 2**(n-1) - (max_correlation // 2)
            bic_nl_values.append(nl)
    return min(bic_nl_values), bic_nl_values

# Bit Independence Criterion—Strict Avalanche Criterion (BIC-SAC) Function
def compute_bic_sac(binary_sbox, n, m):
    bic_sac_values = []
    for i in range(m):
        for j in range(i + 1, m):
            sac_matrix = np.zeros(n)
            for k in range(n):
                count = 0
                for x in range(2**n):
                    flipped_x = x ^ (1 << k)
                    original = binary_sbox[x, i] ^ binary_sbox[x, j]
                    flipped = binary_sbox[flipped_x, i] ^ binary_sbox[flipped_x, j]
                    if original != flipped:
                        count += 1
                sac_matrix[k] = count / (2**n)
            bic_sac_values.append(np.mean(sac_matrix))
    return np.mean(bic_sac_values), bic_sac_values

# Linear Approximation Probability (LAP) Function
def compute_lap(sbox, n):
    bias_table = np.zeros((2**n, 2**n))
    for a in range(1, 2**n):
        for b in range(2**n):
            count = 0
            for x in range(2**n):
                input_dot = bin(x & a).count('1') % 2
                output_dot = bin(sbox[x] & b).count('1') % 2
                if input_dot == output_dot:
                    count += 1
            bias_table[a, b] = abs(count - 2**(n-1))
    lap_max = np.max(bias_table) / (2**n)
    return lap_max, bias_table

# Differential Approximation Probability (DAP) Function
def compute_dap(sbox, n):
    diff_table = np.zeros((2**n, 2**n), dtype=int)
    for delta_x in range(1, 2**n):
        for x in range(2**n):
            x_prime = x ^ delta_x
            delta_y = sbox[x] ^ sbox[x_prime]
            diff_table[delta_x, delta_y] += 1
    dap_max = np.max(diff_table) / (2**n)
    return dap_max, diff_table

# main Function
def main():
    # page 
    st.set_page_config(
    page_title="S-Box Analyzer",
    page_icon="https://raw.githubusercontent.com/risdaaaa/S-Box-Explorer/main/s-box-image.png",
    layout="wide"
)

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div style="background-color: #f7f7f7; padding: 20px; border-radius: 10px;">
                <h1 style="font-size: 26px; color: #2E86C1; text-align: center;">🛠️ S-Box Analyzer</h1>
                <div style="text-align: center; margin: 10px 0;">
                    <img src="https://raw.githubusercontent.com/risdaaaa/S-Box-Explorer/main/s-box-image.png" width="200" style="border-radius: 10px;"/>
                </div>
                <p style="font-size: 18px; color: #555; text-align: center;">
                    Upload your file, select the analysis, and download the results.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")

        st.markdown(
            """
            <div style="padding: 10px; background-color: #eef2f3; border-radius: 10px; margin-top: 10px;">
                <h3 style="font-size: 16px; color: #34495E;">📤 Upload S-Box File</h3>
                <p style="font-size: 14px; color: #555;">Accepted formats: Excel (.xlsx, .xls)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader(
            label="", 
            type=["xlsx", "xls"], 
            label_visibility="collapsed"
        )

    # main menu
    st.markdown(
        """
        <div style="background-color: #f9f9f9; padding: 30px; border-radius: 15px;">
            <h1 style="color: #007BFF; font-size: 36px; font-weight: bold; text-align: center; margin-bottom: 20px;">
                🛠️ S-Box Explorer 🛠️
            </h1>
            <div style="text-align: center; font-size: 28px; font-weight: bold; color: #2E86C1; margin-bottom: 10px;">
                Welcome to the <span style="color: #D8C4B6;">S-Box Explorer</span>!
            </div>
            <p style="font-size: 20px; color: #34495E; text-align: center;">
                Discover and analyze the cryptographic properties of S-Boxes effortlessly with this advanced tool.
            </p>
            <div style="font-size: 20px; font-weight: bold; color: #5D6D7E; margin-top: 30px; margin-bottom: 20px;">
                Key Features:
            </div>
            <ul style="font-size: 18px; line-height: 1.8; color: #5D6D7E; padding-left: 40px;">
                <li><b>Nonlinearity (NL)</b> – Measure the S-Box's resilience against linear attacks.</li>
                <li><b>Strict Avalanche Criterion (SAC)</b> – Evaluate output sensitivity to input changes.</li>
                <li><b>Bit Independence Criterion—Nonlinearity (BIC-NL)</b> – Analyze nonlinear relationships between bits.</li>
                <li><b>Bit Independence Criterion—Strict Avalanche Criterion (BIC-SAC)</b> – Assess inter-bit influence on output distribution.</li>
                <li><b>Linear Approximation Probability (LAP)</b> – Ensure robustness against linear approximation attacks.</li>
                <li><b>Differential Approximation Probability (DAP)</b> – Identify resistance to differential attacks.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    

    # if upload file
    if uploaded_file:
        # file and table
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("<h3>📋 Imported S-Box</h3>", unsafe_allow_html=True)
            df = pd.read_excel(uploaded_file, header=None)
            st.dataframe(df, height=300)

        sbox = df.values.flatten()
        n, m = 8, 8  # Default bit input/output
        binary_sbox = sbox_to_binary(sbox, m)

        # sidebar option
        with st.sidebar:
            st.markdown("<h3 style='color: #007BFF;'>⚙️ Analysis Options</h3>", unsafe_allow_html=True)
            test_options = [
                "Nonlinearity (NL)",
                "Strict Avalanche Criterion (SAC)",
                "Bit Independence Criterion-Nonlinearity (BIC-NL)",
                "Bit Independence Criterion-Strict Avalanche (BIC-SAC)",
                "Linear Approximation Probability (LAP)",
                "Differential Approximation Probability (DAP)"
            ]
            selected_tests = st.multiselect("Select the analysis tests:", test_options)

        # Progress bar
        progress_bar = st.progress(0)
        results = {}

        # klik option
        total_tests = len(selected_tests)
        for i, test in enumerate(selected_tests):
            st.markdown(f"<p style='color: #34495E;'>Running <b>{test}</b>...</p>", unsafe_allow_html=True)

            if test == "Nonlinearity (NL)":
                nl_min, nl_values = compute_nl(binary_sbox, n, m)
                results["Nonlinearity (NL)"] = {
                    "Minimum NL": float(nl_min),
                    "NL Per Component Function": [float(val) for val in nl_values]
                }

            elif test == "Strict Avalanche Criterion (SAC)":
                sac_avg, sac_matrix = compute_sac(binary_sbox, n, m)
                results["Strict Avalanche Criterion (SAC)"] = {
                    "Average SAC": float(sac_avg),
                    "SAC Matrix": sac_matrix.astype(float)
                }

            elif test == "Bit Independence Criterion-Nonlinearity (BIC-NL)":
                bic_nl_min, bic_nl_values = compute_bic_nl(binary_sbox, n, m)
                results["Bit Independence Criterion-Nonlinearity (BIC-NL)"] = {
                    "Minimum BIC NL": float(bic_nl_min),
                    "All BIC NL Pairs": [float(val) for val in bic_nl_values]
                }

            elif test == "Bit Independence Criterion-Strict Avalanche (BIC-SAC)":
                bic_sac_avg, bic_sac_values = compute_bic_sac(binary_sbox, n, m)
                results["Bit Independence Criterion-Strict Avalanche (BIC-SAC)"] = {
                    "Average BIC SAC": float(bic_sac_avg),
                    "All BIC SAC Pairs": [float(val) for val in bic_sac_values]
                }

            elif test == "Linear Approximation Probability (LAP)":
                lap_max, bias_table = compute_lap(sbox, n)
                results["Linear Approximation Probability (LAP)"] = {
                    "Maximum LAP": float(lap_max),
                    "Bias Table": bias_table.astype(float)
                }

            elif test == "Differential Approximation Probability (DAP)":
                dap_max, diff_table = compute_dap(sbox, n)
                results["Differential Approximation Probability (DAP)"] = {
                    "Maximum DAP": float(dap_max),
                    "Differential Table": diff_table.astype(float)
                }

            progress_bar.progress((i + 1) / total_tests)

        # analysis result
        st.markdown("<h3>📊 Analysis Results</h3>", unsafe_allow_html=True)
        for test, result in results.items():
            st.markdown(f"<h4 style='color: #2E86C1;'>{test} Results</h4>", unsafe_allow_html=True)
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    st.markdown(f"<p style='font-weight: bold;'>{key}</p>", unsafe_allow_html=True)
                    st.dataframe(value, height=200)
                else:
                    st.markdown(f"<p><b>{key}:</b> {value}</p>", unsafe_allow_html=True)

        # Export results to Excel
        if results:
            # Subheader 
            st.markdown(
                """
                <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-top: 20px; text-align: center;">
                    <h2 style="color: #007BFF; font-size: 24px; font-weight: bold;">
                        📥 Download Your Analysis Results
                    </h2>
                    <p style="font-size: 16px; color: #555; margin-top: 10px;">
                        Your cryptographic analysis is complete! Click the button below to download the results in an Excel file.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # crate file Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for test, result in results.items():
                    for key, value in result.items():
                        if isinstance(value, np.ndarray) or isinstance(value, list):
                            if isinstance(value, list):
                                value = np.array(value)
                            if value.ndim == 1:  # 1D Array
                                pd.DataFrame(value, columns=[key]).to_excel(writer, sheet_name=f"{test}_{key}", index=False)
                            elif value.ndim == 2:  # 2D Array
                                pd.DataFrame(value).to_excel(writer, sheet_name=f"{test}_{key}", index=False, header=False)
                        else:
                            pd.DataFrame({key: [value]}).to_excel(writer, sheet_name=f"{test}_{key}", index=False)

            # download file
            st.markdown(
                """
                <div style="text-align: center; margin-top: 20px;">
                    <a href="data:application/vnd.ms-excel;base64,{encoded_excel}" download="sbox_analysis_results.xlsx" style="
                        display: inline-block;
                        background-color: #007BFF;
                        color: #ffffff;
                        padding: 12px 20px;
                        font-size: 16px;
                        font-weight: bold;
                        text-decoration: none;
                        border-radius: 5px;
                        transition: background-color 0.3s ease;">
                        📥 Download Results
                    </a>
                </div>
                """.format(encoded_excel=output.getvalue().hex()),
                unsafe_allow_html=True,
            )


    else:
        # upload ur file
        st.markdown(
            """
            <div style="background-color: #d9f7be; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #a6cf82;">
                <h3 style="color: #3e8e41; font-size: 20px; font-weight: bold;">
                    📂 <span style="color: #007BFF;">Please upload your S-Box Excel file</span> to get started with the analysis!
                </h3>
                <p style="color: #555; font-size: 16px; margin-top: 10px;">
                    Simply click the "Upload" button below and we'll handle the rest. Let's explore the cryptographic properties of your S-Box!
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()