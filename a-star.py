import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from queue import PriorityQueue
import random
from IPython.display import HTML, display  # Adicionando importação do display

# Configurações do labirinto
TAMANHO = 32
INICIO = (1, 1)
FIM = (TAMANHO-2, TAMANHO-2)
CELULA_TAMANHO_CM = 10  # Cada célula equivale a 10 cm

# Cores para visualização
CORES = {
    'parede': '#000000',     # Preto
    'caminho': '#FFFFFF',    # Branco
    'inicio': '#00FF00',     # Verde
    'fim': '#FF0000',        # Vermelho
    'explorado': '#ADD8E6',  # Azul claro
    'caminho_final': '#FFFF00'  # Amarelo
}

def criar_labirinto(tamanho, densidade_paredes=0.25):
    """Cria um labirinto com paredes e caminhos aleatórios"""
    labirinto = np.zeros((tamanho, tamanho))
    
    # Preenche o labirinto com paredes nas bordas
    labirinto[0, :] = 1
    labirinto[-1, :] = 1
    labirinto[:, 0] = 1
    labirinto[:, -1] = 1
    
    # Adiciona paredes aleatórias no interior
    for i in range(1, tamanho-1):
        for j in range(1, tamanho-1):
            if random.random() < densidade_paredes and (i, j) != INICIO and (i, j) != FIM:
                labirinto[i, j] = 1
                
    return labirinto

def heuristica_manhattan(p1, p2):
    """Calcula a distância de Manhattan entre dois pontos"""
    x1, y1 = p1
    x2, y2 = p2
    return (abs(x1 - x2) + abs(y1 - y2)) * CELULA_TAMANHO_CM

def vizinhos_validos(labirinto, ponto):
    """Retorna os vizinhos válidos de um ponto (não são paredes)"""
    x, y = ponto
    vizinhos = []
    
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Movimentos: direita, baixo, esquerda, cima
        nx, ny = x + dx, y + dy
        
        # Verifica se está dentro dos limites e não é parede
        if 0 <= nx < labirinto.shape[0] and 0 <= ny < labirinto.shape[1] and labirinto[nx, ny] == 0:
            vizinhos.append((nx, ny))
            
    return vizinhos

def a_estrela(labirinto, inicio, fim):
    """Implementação do algoritmo A*"""
    fronteira = PriorityQueue()
    fronteira.put((0, inicio))
    
    veio_de = {inicio: None}
    custo_ate_agora = {inicio: 0}
    
    explorados = []  # Para armazenar a ordem de exploração
    
    while not fronteira.empty():
        atual = fronteira.get()[1]
        explorados.append(atual)
        
        if atual == fim:
            break
            
        for vizinho in vizinhos_validos(labirinto, atual):
            novo_custo = custo_ate_agora[atual] + CELULA_TAMANHO_CM  # Custo uniforme de 10 cm por célula
            
            if vizinho not in custo_ate_agora or novo_custo < custo_ate_agora[vizinho]:
                custo_ate_agora[vizinho] = novo_custo
                prioridade = novo_custo + heuristica_manhattan(vizinho, fim)
                fronteira.put((prioridade, vizinho))
                veio_de[vizinho] = atual
                
    # Reconstruir o caminho
    caminho = []
    atual = fim
    
    while atual != inicio:
        caminho.append(atual)
        atual = veio_de.get(atual, None)
        if atual is None:
            return None, explorados  # Não há caminho
        
    caminho.append(inicio)
    caminho.reverse()
    
    return caminho, explorados

def atualizar_visualizacao(frame, labirinto, caminho, explorados, img):
    """Atualiza a visualização para cada frame da animação"""
    # Cria uma matriz de cores
    visual = np.zeros((labirinto.shape[0], labirinto.shape[1], 3))
    
    for i in range(labirinto.shape[0]):
        for j in range(labirinto.shape[1]):
            if labirinto[i, j] == 1:
                visual[i, j] = mcolors.to_rgb(CORES['parede'])
            else:
                visual[i, j] = mcolors.to_rgb(CORES['caminho'])
    
    # Marca início e fim
    visual[INICIO[0], INICIO[1]] = mcolors.to_rgb(CORES['inicio'])
    visual[FIM[0], FIM[1]] = mcolors.to_rgb(CORES['fim'])
    
    # Marca células exploradas até o frame atual
    if explorados and frame < len(explorados):
        for i in range(frame + 1):
            x, y = explorados[i]
            if (x, y) != INICIO and (x, y) != FIM:
                visual[x, y] = mcolors.to_rgb(CORES['explorado'])
    
    # Marca o caminho final no último frame
    if caminho and frame == len(explorados):
        for x, y in caminho:
            if (x, y) != INICIO and (x, y) != FIM:
                visual[x, y] = mcolors.to_rgb(CORES['caminho_final'])
    
    img.set_array(visual)
    plt.title(f"Passo {frame}/{len(explorados) if explorados else 0}")
    return img

def animar_solucao_otimizada(labirinto, caminho, explorados):
    """Versão otimizada da animação que mostra apenas 1/10 dos frames"""
    fig, ax = plt.subplots(figsize=(8, 8))  # Tamanho reduzido
    
    # Cria a imagem inicial
    visual = np.zeros((labirinto.shape[0], labirinto.shape[1], 3))
    img = ax.imshow(visual)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Mostra apenas 1/10 dos frames para reduzir o tamanho
    frames_para_mostrar = explorados[::10] + ([caminho] if caminho else [])
    total_frames = len(frames_para_mostrar)
    
    def update(frame_idx):
        ax.clear()
        frame = frames_para_mostrar[frame_idx]
        
        # Cria a visualização para este frame
        visual = np.zeros((labirinto.shape[0], labirinto.shape[1], 3))
        for i in range(labirinto.shape[0]):
            for j in range(labirinto.shape[1]):
                if labirinto[i, j] == 1:
                    visual[i, j] = mcolors.to_rgb(CORES['parede'])
                else:
                    visual[i, j] = mcolors.to_rgb(CORES['caminho'])
        
        visual[INICIO[0], INICIO[1]] = mcolors.to_rgb(CORES['inicio'])
        visual[FIM[0], FIM[1]] = mcolors.to_rgb(CORES['fim'])
        
        # Marca células exploradas
        if frame_idx < len(frames_para_mostrar)-1:
            for x, y in frames_para_mostrar[:frame_idx+1]:
                if (x, y) != INICIO and (x, y) != FIM:
                    visual[x, y] = mcolors.to_rgb(CORES['explorado'])
        else:
            # Mostra o caminho final
            if caminho:
                for x, y in caminho:
                    if (x, y) != INICIO and (x, y) != FIM:
                        visual[x, y] = mcolors.to_rgb(CORES['caminho_final'])
        
        ax.imshow(visual)
        ax.set_title(f"Passo {frame_idx*10}/{len(explorados)}" if frame_idx < len(frames_para_mostrar)-1 
                    else f"Caminho Final ({len(caminho)} passos)")
        return ax

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames_para_mostrar),
        interval=100,  # Intervalo maior
        repeat=False
    )
    
    plt.close()
    return ani

# Cria o labirinto menor
labirinto = criar_labirinto(TAMANHO, densidade_paredes=0.25)





# Encontra o caminho
caminho, explorados = a_estrela(labirinto, INICIO, FIM)

if caminho:
    print(f"Caminho encontrado com {len(caminho)} passos!")
    print(f"Células exploradas: {len(explorados)}")
    
    try:
        # Tenta mostrar no Jupyter
        from IPython.display import HTML
        animacao = animar_solucao_otimizada(labirinto, caminho, explorados)
        plt.rcParams['animation.embed_limit'] = 50.0
        display(HTML(animacao.to_html5_video()))
    except:
        # Fallback para salvar como arquivo
        print("Salvando animação como arquivo...")
        try:
            animacao.save('labirinto_a_star.mp4', writer=FFMpegWriter(fps=10))
            print("Salvo como labirinto_a_star.mp4")
        except:
            try:
                animacao.save('labirinto_a_star.gif', writer=PillowWriter(fps=10))
                print("Salvo como labirinto_a_star.gif")
            except Exception as e:
                print(f"Erro ao salvar animação: {e}")
    
    