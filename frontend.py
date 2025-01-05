import streamlit as st
import requests

# FastAPI URL'si
API_URL = "http://127.0.0.1:8000/ask"

# Başlık ve açıklama
st.title("💬 Soru-Cevap Sistemi - Chat Akışı")
st.write("📘 **FastAPI üzerinden çalışan bir chatbot. Sorularınızı sorun ve anında yanıt alın.**")

# Sohbet geçmişi için bir session_state tanımlıyoruz
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # Sohbet geçmişi için liste

# Kullanıcıdan soru al
with st.form("question_form"):
    question = st.text_input("Sorunuzu yazın:", key="user_input")
    submitted = st.form_submit_button("Gönder")

# Kullanıcı soru gönderdiğinde API'ye istek gönder
if submitted:
    if question.strip() == "":
        st.warning("⚠️ Lütfen bir soru girin.")
    else:
        try:
            # API'ye POST isteği gönder
            response = requests.post(API_URL, json={"question": question})
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "Yanıt alınamadı.")
                
                # Sohbet geçmişine ekle
                st.session_state["chat_history"].append({"user": question, "bot": answer})
            else:
                st.error(f"❌ Bir hata oluştu: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"⚠️ API bağlantısında bir sorun oluştu: {e}")

# Sohbet geçmişini görüntüle
st.write("### Sohbet Geçmişi:")
for chat in st.session_state["chat_history"]:
    st.markdown(f"**👤 Kullanıcı:** {chat['user']}")
    st.markdown(f"**🤖 Bot:** {chat['bot']}")
    st.write("---")
