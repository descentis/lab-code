import psutil
import streamlit as st
import time


def get_resource_usage():
    # Get memory and CPU usage
    memory = psutil.virtual_memory()
    cpu_percentage = psutil.cpu_percent(interval=1)
    disk = psutil.disk_usage('/')

    return {
        "total_memory": memory.total / (1024 ** 2),  # MB
        "used_memory": memory.used / (1024 ** 2),  # MB
        "memory_percent": memory.percent,
        "cpu_percent": cpu_percentage,
        "total_disk": disk.total / (1024 ** 3),  # GB
        "used_disk": disk.used / (1024 ** 3),  # GB
        "free_disk": disk.free / (1024 ** 3),  # GB
        "disk_percent": disk.percent,
    }


def display_resource_usage():
    # Initialize session state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = 0

    # Display the resource usage
    current_time = time.time()
    if current_time - st.session_state.last_refresh > 5:  # Refresh every 5 seconds
        st.session_state.resources = get_resource_usage()
        st.session_state.last_refresh = current_time

    # Display metrics
    st.sidebar.markdown("<h1>Resource Usage</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3>Memory Usage (MB)</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<p style='font-size: 15px;'>Total Memory : {st.session_state.resources['total_memory']:.2f} MB</p>",
        unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<p style='font-size: 15px;'>Used Memory : {st.session_state.resources['used_memory']:.2f} MB</p>",
        unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<p style='font-size: 15px;'>Memory Usage : {st.session_state.resources['memory_percent']:.2f}%</p>",
        unsafe_allow_html=True)

    st.sidebar.markdown("<h3>CPU Usage (%)</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<p style='font-size: 15px;'>CPU Usage : {st.session_state.resources['cpu_percent']:.2f}%</p>",
        unsafe_allow_html=True)

    st.sidebar.markdown("<h3>Disk Usage (GB)</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<p style='font-size: 15px;'>Total Disk : {st.session_state.resources['total_disk']:.2f} GB</p>",
        unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<p style='font-size: 15px;'>Used Disk : {st.session_state.resources['used_disk']:.2f} GB</p>",
        unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<p style='font-size: 15px;'>Free Disk : {st.session_state.resources['free_disk']:.2f} GB</p>",
        unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<p style='font-size: 15px;'>Disk Usage : {st.session_state.resources['disk_percent']:.2f}%</p>",
        unsafe_allow_html=True)
