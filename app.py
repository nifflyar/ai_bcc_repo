import streamlit as st
import pandas as pd
import os
import shutil
from analyzer import process_case_folder

st.set_page_config(page_title="Анализ клиентов", layout="wide")
st.title("Анализ поведения клиентов и предложения им офферов")

mode = st.radio("Выберите режим:", ["Все клиенты", "Один клиент"])

clients_file = st.file_uploader("Загрузите clients.csv", type="csv")
tx_folder = st.text_input("Укажите путь к папке transactions", "./case1/")
tr_folder = st.text_input("Укажите путь к папке transfers", "./case1/")

out_folder = "./out"
os.makedirs(out_folder, exist_ok=True)
out_csv = os.path.join(out_folder, "push_output.csv")
meta_jsonl = os.path.join(out_folder, "meta_per_client.jsonl")

if mode == "Один клиент":
    client_id = st.number_input("Введите client_code", min_value=1, step=1)

if st.button("Запустить анализ"):
    if not clients_file or not os.path.isdir(tx_folder) or not os.path.isdir(tr_folder):
        st.error("Загрузите clients.csv и укажите корректные пути к transactions и transfers")
    else:
        case_folder = "./case_tmp"
        os.makedirs(case_folder, exist_ok=True)


        clients_path = os.path.join(case_folder, "clients.csv")
        with open(clients_path, "wb") as f:
            f.write(clients_file.getbuffer())

        if mode == "Все клиенты":

            shutil.copytree(tx_folder, os.path.join(case_folder, "transactions"), dirs_exist_ok=True)
            shutil.copytree(tr_folder, os.path.join(case_folder, "transfers"), dirs_exist_ok=True)

            products = process_case_folder(case_folder, out_csv, meta_jsonl)

            st.success("Анализ завершён")
            if os.path.exists(out_csv):
                df = pd.read_csv(out_csv)
                st.subheader("Результаты по клиентам")
                st.dataframe(df)

                st.download_button("⬇️ Скачать результаты", df.to_csv(index=False).encode("utf-8"),
                                   "push_results.csv", "text/csv")

        elif mode == "Один клиент":
            tx_file = os.path.join(tx_folder, f"client_{client_id}_transactions_3m.csv")
            tr_file = os.path.join(tr_folder, f"client_{client_id}_transfers_3m.csv")

            if not os.path.exists(tx_file) or not os.path.exists(tr_file):
                st.error(f"Не найдены файлы для клиента {client_id}")
            else:

                os.makedirs(os.path.join(case_folder, "transactions"), exist_ok=True)
                os.makedirs(os.path.join(case_folder, "transfers"), exist_ok=True)

                shutil.copy(tx_file, os.path.join(case_folder, "transactions"))
                shutil.copy(tr_file, os.path.join(case_folder, "transfers"))

                products = process_case_folder(case_folder, out_csv, meta_jsonl)

                st.success(f"Анализ клиента {client_id} завершён ")
                if os.path.exists(out_csv):
                    df = pd.read_csv(out_csv)
                    st.subheader("Результат")
                    st.dataframe(df[df["client_code"] == client_id])
