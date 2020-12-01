# E: 빈 공간(Empty) = 0, X: 선공 Player = 1, O: 후공 Player = 2

class TicTacToe:
    def __init__(self):
        self.N = 3  #: 정방행렬의 변 길이
        self.map = [[0 for _ in range(self.N)] for _ in range(self.N)]
        self.map_index_description = [h * self.N + w for h in range(self.N) for w in range(self.N)]
        self.player_types = ('X', 'O')  # (X, O)
        self.global_step = 0

        self.win_reward = 1.0
        self.defeat_reward = -1.0
        self.draw_reward = 0.0
        self.player_result = {'X': self.draw_reward, 'O': self.draw_reward}

        self.done = False

    # game.reset()
    def reset(self):
        self.map = [[0 for _ in range(self.N)] for _ in range(self.N)]
        self.global_step = 0
        self.player_result = {'X': self.draw_reward, 'O': self.draw_reward}
        self.done = False

        return self.map

    # game.step(0~8)
    def step(self, action):
        action_coord_h, action_coord_w = self.transform_action(action)
        if self.global_step % 2 == 0:
            current_player_idx = 0
            other_player_idx = 1
        else:
            current_player_idx = 1
            other_player_idx = 0
        current_player_type = self.player_types[current_player_idx]
        other_player_type = self.player_types[other_player_idx]

        # 현재 플레이어가 착수를 할 위치가 가능한 곳인가?
        if self.map[action_coord_h][action_coord_w] == 0:
            current_player = self.player_types.index(current_player_type) + 1
            self.map[action_coord_h][action_coord_w] = self.player_types.index(current_player_type) + 1
            # 승부의 결과?
            if self.is_win(current_player):  # 현재 플레이어 승리
                self.player_result[current_player_type] = self.win_reward
                self.player_result[other_player_type] = self.defeat_reward
                self.done = True
            elif self.is_full():  # 무승부
                self.done = True
            else:
                pass
        else:  # 현재 플레이어 패배
            self.player_result[current_player_type] = self.defeat_reward
            self.player_result[other_player_type] = self.win_reward
            self.done = True
        self.global_step += 1

        return self.map, self.player_result, self.done

    def transform_action(self, action):
        return divmod(action, self.N)

    def is_win(self, current_player_type):
        vertical_win = [True for _ in range(self.N)]
        horizontal_win = [True for _ in range(self.N)]
        diagonal_win = [True for _ in range(2)]
        for h in range(self.N):
            for w in range(self.N):
                # 가로, 세로
                if self.map[h][w] != current_player_type:
                    vertical_win[h] = False
                    horizontal_win[w] = False
                else:
                    pass
                # 왼 대각
                if h == w and self.map[h][w] != current_player_type:
                    diagonal_win[0] = False
                # 오른 대각
                rotated_w = abs(w - (self.N - 1))
                if h == rotated_w and self.map[h][w] != current_player_type:
                    diagonal_win[1] = False
        if any(vertical_win) or any(horizontal_win) or any(diagonal_win):
            return True
        else:
            return False

    def is_full(self):
        for h in range(self.N):
            for w in range(self.N):
                if self.map[h][w] == 0:
                    return False
                else:
                    pass
        return True

    def print_description(self):
        print("** Initial NxN Tic-tac-toe Map **")
        self.print_current_map()

        print("** Action Indexes **")
        for idx, des in enumerate(self.map_index_description):
            print(des, end=' ')
            if (idx + 1) % self.N == 0:
                print('\n', end='')

    def print_current_map(self):
        for h in range(self.N):
            for w in range(self.N):
                print(self.map[h][w], end=' ')
            print('\n', end='')
        print()

    # Fill this function
    def match_prediction(self):
        # X: 현재 플레이어
        # return value (이긴다: 1 무승부: 0 진다: -1)
        # print("'X'의 최대 결과는 '승리'입니다.")
        # print("'O'의 최대 결과는 '패배'입니다.")
        # print("현재 플레이어의 최대 결과는 패배입니다.")
        print("현재 착수할 플레이어가 이긴다고 확신이 드는가? Yes or No")


if __name__ == '__main__':
    game = TicTacToe()
    game.print_description()

    game.reset()
    done = False
    while not done:
        print()

        action = int(input('Select action please: '))
        if not (game.map_index_description[0] <= action <= game.map_index_description[-1]):
            done = True
            print("Error: You entered the wrong number.")
            continue

        _, player_result, done = game.step(action)
        game.print_current_map()
        if done:
            for player, result in player_result.items():
                if result == game.win_reward:
                    player_result[player] = 'win'
                elif result == game.defeat_reward:
                    player_result[player] = 'defeat'
                else:
                    player_result[player] = 'draw'
            print(player_result)