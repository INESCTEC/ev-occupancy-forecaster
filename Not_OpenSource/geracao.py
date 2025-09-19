import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def gerar_ocupacao_txt(
    output_path="ocupacao.txt",
    inicio="2023-01-01 00:00:00",
    fim="2024-12-31 23:55:00",
    freq="5min",
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    # Criar todos os timestamps de 5 em 5 minutos
    timestamps = pd.date_range(start=inicio, end=fim, freq=freq)
    ocupacoes = []

    for dia in pd.date_range(start=inicio, end=fim, freq="D"):
        hora = dia.replace(hour=0, minute=0)
        bloco_dia = []

        while hora < dia + timedelta(days=1):
            hora_str = hora.strftime("%H:%M")
            mes = hora.month

            # Verão: junho, julho, agosto → mais "0"
            is_verao = mes in [6, 7, 8]

            if 8 <= hora.hour < 20:
                # Dentro do horário de ocupação
                if is_verao:
                    prob_ocupado = 0.6
                else:
                    prob_ocupado = 0.9
            else:
                # Fora do horário de ocupação
                prob_ocupado = 0.05

            # Gerar blocos de pelo menos 4 passos (20min)
            duracao_bloco = random.randint(4, 20)
            valor = 1 if random.random() < prob_ocupado else 0

            for _ in range(duracao_bloco):
                if hora > dia + timedelta(days=1) - timedelta(minutes=5):
                    break
                bloco_dia.append((hora, valor))
                hora += timedelta(minutes=5)

        ocupacoes.extend(bloco_dia)

    # Guardar no ficheiro
    with open(output_path, "w", encoding="utf-8") as f:
        for ts, val in ocupacoes:
            linha = f"{ts.strftime('%Y-%m-%d %H:%M:%S')}\t{val}\n"
            f.write(linha)

    print(f"Ficheiro gerado com sucesso: {output_path}")
    print(f"Total de linhas: {len(ocupacoes)}")


# Execução principal
if __name__ == "__main__":
    gerar_ocupacao_txt()
