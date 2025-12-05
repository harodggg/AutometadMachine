import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class CellularAutomaton:
    def __init__(self, size=100, probability=0.2):
        self.size = size
        # 初始化网格：随机生成 0 或 1
        # choice([0, 1]) 模拟死或生
        self.grid = np.random.choice(
            [0, 1], size*size, p=[1-probability, probability]).reshape(size, size)

    def update(self, frameNum, img, grid, size):
        # 复制一份当前网格，避免计算干扰
        newGrid = grid.copy()

        # --- 核心逻辑：向量化计算邻居数量 ---
        # 利用 np.roll 进行位移，相当于卷积核 (Convolution) 的操作
        # 这比写两层 for 循环快几百倍，逻辑类似于 CNN 的 Pooling

        # 上下左右
        n_up = np.roll(grid, -1, axis=0)
        n_down = np.roll(grid,  1, axis=0)
        n_left = np.roll(grid, -1, axis=1)
        n_right = np.roll(grid,  1, axis=1)

        # 对角线
        n_ul = np.roll(n_up, -1, axis=1)    # 上左
        n_ur = np.roll(n_up,  1, axis=1)    # 上右
        n_dl = np.roll(n_down, -1, axis=1)  # 下左
        n_dr = np.roll(n_down,  1, axis=1)  # 下右

        # 计算每个细胞周围活着的邻居总数
        neighbors = n_up + n_down + n_left + n_right + n_ul + n_ur + n_dl + n_dr

        # --- 应用规则 (冯·诺伊曼思想的简化) ---

        # 规则 1: 孤独死 (邻居 < 2) 或 拥挤死 (邻居 > 3)
        # (grid == 1) & ((neighbors < 2) | (neighbors > 3))
        newGrid[(grid == 1) & ((neighbors < 2) | (neighbors > 3))] = 0

        # 规则 2: 复活 (死细胞周围正好有 3 个邻居)
        # (grid == 0) & (neighbors == 3)
        newGrid[(grid == 0) & (neighbors == 3)] = 1

        # 规则 3: 保持 (活细胞周围有 2 或 3 个邻居) -> 此时值不变，无需操作

        # 更新数据
        img.set_data(newGrid)
        grid[:] = newGrid[:]
        return img,

    def run(self):
        # 设置可视化
        fig, ax = plt.subplots()
        img = ax.imshow(self.grid, interpolation='nearest', cmap='inferno')
        ax.set_title("Von Neumann's Legacy: Cellular Automata (Vectorized)")
        plt.axis('off')

        # 启动动画
        ani = animation.FuncAnimation(fig, self.update, fargs=(img, self.grid, self.size),
                                      frames=10, interval=50, save_count=50)
        plt.show()


if __name__ == '__main__':
    ca = CellularAutomaton(size=100, probability=0.2)
    ca.run()
