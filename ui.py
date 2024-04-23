import streamlit as st
import base64

@st.cache_data
def displayPDF(file):
    with open(file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        # Embed PDF in HTML
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        
        # Display File
        st.markdown(pdf_display, unsafe_allow_html=True)