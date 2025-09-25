import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# App Settings
# -----------------------------
st.set_page_config(page_title="📊 EMG + ROM Dashboard", layout="wide")
st.title("📊 EMG + ROM Dashboard (Interactive)")

# Fixed sampling rate (Hz)
fs = 50.0   # 50 samples per second

# -----------------------------
# File Upload
# -----------------------------
st.subheader("📂 Upload Patient Session Data")
uploaded_emg = st.file_uploader("Upload EMG data (.txt)", type=["txt"])
uploaded_rom = st.file_uploader("Upload ROM data (.txt)", type=["txt"])

if uploaded_emg is not None and uploaded_rom is not None:
    # -----------------------------
    # Load EMG
    # -----------------------------
    raw_text_emg = uploaded_emg.read().decode("utf-8").strip()
    tokens_emg = raw_text_emg.replace("\n", ",").replace(" ", ",").split(",")
    emg_vals = np.array([float(x) for x in tokens_emg if x.strip() != ""])

    # Load ROM
    raw_text_rom = uploaded_rom.read().decode("utf-8").strip()
    tokens_rom = raw_text_rom.replace("\n", ",").replace(" ", ",").split(",")
    rom_vals = np.array([float(x) for x in tokens_rom if x.strip() != ""])

    # Match length (truncate to shorter signal)
    n = min(len(emg_vals), len(rom_vals))
    emg_vals = emg_vals[:n]
    rom_vals = rom_vals[:n]
    time = np.arange(n) / fs

    st.success(f"✅ Loaded EMG ({len(emg_vals)} samples) and ROM ({len(rom_vals)} samples)")
    st.write(f"**Recording duration:** {n/fs:.2f} seconds at {fs} Hz")

    # -----------------------------
    # Parameters
    # -----------------------------
    st.sidebar.header("⚙️ Parameters")
    show_emg = st.sidebar.checkbox("Show EMG", value=True)
    show_env = st.sidebar.checkbox("Show RMS Envelope (EMG)", value=True)
    show_rom = st.sidebar.checkbox("Show ROM", value=True)

    env_win = st.sidebar.slider(
        "Envelope Window (samples)", 5, 500, 50, 5,
        help="Number of samples for RMS smoothing. At 50 Hz, 50 samples = 1 sec."
    )
    thresh_mult = st.sidebar.slider(
        "Burst Threshold Multiplier", 0.1, 5.0, 1.5, 0.1,
        help="Multiplier × std added to mean envelope to set burst threshold."
    )

    # -----------------------------
    # EMG Envelope & Bursts
    # -----------------------------
    bursts = []
    rms_env = None
    thresh = None

    if show_env:
        sq = emg_vals**2
        csum = np.cumsum(np.insert(sq, 0, 0))
        rms_env = np.sqrt((csum[env_win:] - csum[:-env_win]) / env_win)
        rms_env = np.concatenate([np.full(env_win-1, np.nan), rms_env])

        env_mean = np.nanmean(rms_env)
        env_std = np.nanstd(rms_env)
        thresh = env_mean + thresh_mult * env_std

        is_burst = rms_env > thresh
        in_burst = False
        for i, b in enumerate(is_burst):
            if b and not in_burst:
                start = i
                in_burst = True
            elif not b and in_burst:
                end = i - 1
                bursts.append((start, end))
                in_burst = False
        if in_burst:
            bursts.append((start, n-1))

    # -----------------------------
    # Peak Analysis
    # -----------------------------
    abs_peak_val = float(np.max(emg_vals))
    abs_peak_idx = int(np.argmax(emg_vals))

    # Top 3 peaks (no outlier cleaning)
    peak_indices_sorted = np.argsort(emg_vals)[::-1][:3]
    top3_peaks = [(float(emg_vals[i]), i/fs) for i in peak_indices_sorted]

    # -----------------------------
    # Summary Metrics (Expanded)
    # -----------------------------
    summary = {
        "n_samples": n,
        "duration_sec": n/fs,
        "emg_min": float(np.min(emg_vals)),
        "emg_max (absolute peak)": abs_peak_val,
        "emg_mean": float(np.mean(emg_vals)),
        "emg_median": float(np.median(emg_vals)),
        "emg_std": float(np.std(emg_vals)),
        "emg_RMS": float(np.sqrt(np.mean(emg_vals**2))),
        "rom_min": float(np.min(rom_vals)),
        "rom_max": float(np.max(rom_vals)),
        "rom_mean": float(np.mean(rom_vals)),
        "rom_median": float(np.median(rom_vals)),
        "rom_std": float(np.std(rom_vals)),
        "n_bursts_detected": len(bursts) if show_env else "Envelope OFF",
        "abs_peak_time_s": abs_peak_idx/fs,
    }

    st.subheader("📈 Summary Statistics")
    st.dataframe(pd.DataFrame(summary.items(), columns=["Metric", "Value"]))

    # Show top 3 peaks separately
    st.subheader("🔝 Top 3 EMG Peaks")
    peak_table = pd.DataFrame(top3_peaks, columns=["Peak Value", "Time (s)"])
    st.table(peak_table)

    # -----------------------------
    # Interactive Overlay Plot
    # -----------------------------
    st.subheader("📊 EMG + ROM Graphs")

    fig = go.Figure()

    if show_emg:
        fig.add_trace(go.Scatter(
            x=time, y=emg_vals,
            mode="lines",
            name="EMG (raw)",
            line=dict(color="blue")
        ))
        # Mark top 3 peaks
        for j, (val, t) in enumerate(top3_peaks):
            fig.add_trace(go.Scatter(
                x=[t], y=[val],
                mode="markers+text",
                text=[f"Peak {j+1}"],
                textposition="top center",
                marker=dict(size=10, symbol="triangle-up"),
                name=f"Peak {j+1}"
            ))

    if show_env and rms_env is not None:
        fig.add_trace(go.Scatter(
            x=time, y=rms_env,
            mode="lines",
            name="RMS Envelope",
            line=dict(color="orange")
        ))
        fig.add_trace(go.Scatter(
            x=[time[0], time[-1]],
            y=[thresh, thresh],
            mode="lines",
            name="Burst Threshold",
            line=dict(color="red", dash="dash")
        ))
        for start, end in bursts:
            fig.add_vrect(
                x0=start/fs, x1=end/fs,
                fillcolor="red", opacity=0.2, line_width=0
            )

    if show_rom:
        fig.add_trace(go.Scatter(
            x=time, y=rom_vals,
            mode="lines",
            name="ROM",
            line=dict(color="purple", dash="dot"),
            yaxis="y2"
        ))

    fig.update_layout(
        title="Overlayed EMG + ROM",
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="EMG Amplitude"),
        yaxis2=dict(title="ROM", overlaying="y", side="right"),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Burst Info
    # -----------------------------
    if show_env and rms_env is not None:
        burst_df = pd.DataFrame(bursts, columns=["Start_index", "End_index"])
        burst_df["Start_time (s)"] = burst_df["Start_index"] / fs
        burst_df["End_time (s)"] = burst_df["End_index"] / fs
        burst_df["Duration (s)"] = burst_df["End_time (s)"] - burst_df["Start_time (s)"]

        st.subheader("⚡ Detected Bursts (EMG)")
        st.dataframe(burst_df)

        # -----------------------------
        # Download Results
        # -----------------------------
        st.subheader("⬇️ Download Results")
        burst_csv = burst_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Bursts CSV", burst_csv, "emg_bursts.csv", "text/csv")

        summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
        summary_csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Summary CSV", summary_csv, "emg_summary.csv", "text/csv")

else:
    st.info("👆 Please upload both EMG and ROM `.txt` files (comma, space, or newline separated).")
