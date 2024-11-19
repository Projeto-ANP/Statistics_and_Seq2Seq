import shutil
import time
import os

def delete_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Pasta '{folder_path}' foi deletada.")
        else:
            print(f"Pasta '{folder_path}' não existe.")
    except Exception as e:
        print(f"Falha ao deletar a pasta '{folder_path}'. Motivo: {e}")
        

def main():
    folder_path = './lightning_logs'
    while True:
        delete_folder(folder_path)
        time.sleep(10 * 60)

if __name__ == "__main__":
    main()
