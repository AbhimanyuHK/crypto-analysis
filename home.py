import streamlit as st
from streamlit_flow import streamlit_flow, StreamlitFlowNode, StreamlitFlowEdge, StreamlitFlowState

st.set_page_config(page_title="Crypto Analysis App", layout="wide")

st.title("📊 Crypto Analysis App")
st.markdown("""
Welcome!  
Use the left sidebar to navigate between:
- **Fetch Data**
- **Daily Differences**
- **Threshold Analysis**
- **Prediction**
- **Best Buy Options**
""")

# Define nodes
nodes = [
    StreamlitFlowNode(id="1", data={"label": "Fetch data from Yahoo!"}, pos=(100, 100)),   # Position: (x=100, y=100)
    StreamlitFlowNode(id="2", data={"label": "Daily Difference"}, pos=(300, 100)),  # Position: (x=300, y=100)
    StreamlitFlowNode(id="3", data={"label": "Threshold Analysis"}, pos=(500, 100)),  # Position: (x=500, y=100)
    StreamlitFlowNode(id="4", data={"label": "Prediction by ML"}, pos=(700, 100)),    # Position: (x=700, y=100)
    StreamlitFlowNode(id="5", data={"label": "Best Buy Options"}, pos=(900, 100)),    # Position: (x=700, y=100)
]

# Define edges
edges = [
    StreamlitFlowEdge(id="e1-2", source="1", target="2"),
    StreamlitFlowEdge(id="e2-3", source="2", target="3"),
    StreamlitFlowEdge(id="e3-4", source="3", target="4"),
    StreamlitFlowEdge(id="e4-5", source="4", target="5"),
]

# Create the state object
state = StreamlitFlowState(nodes, edges)

# Use session state so state is preserved across reruns
if 'flow_state' not in st.session_state:
    st.session_state.flow_state = state

# Render (and update) the flow, storing any updated state
st.session_state.flow_state = streamlit_flow('flow', st.session_state.flow_state, height=500)
