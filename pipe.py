# pipe.py: Projeto de Inteligência Artificial 2023/2024.

# Group 47:
# 106987 Raquel Grãos Rodrigues
# 107057 Guilherme Ribeiro Pereira

import sys
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
    depth_limited_search,

)


pieces_dict = {
    "FC": [1, 0, 0, 0],
    "FB": [0, 1, 0, 0],
    "FE": [0, 0, 1, 0],
    "FD": [0, 0, 0, 1],
    "BC": [1, 0, 1, 1],
    "BB": [0, 1, 1, 1],
    "BE": [1, 1, 1, 0],
    "BD": [1, 1, 0, 1],
    "VC": [1, 0, 1, 0],
    "VB": [0, 1, 0, 1],
    "VE": [0, 1, 1, 0],
    "VD": [1, 0, 0, 1],
    "LH": [0, 0, 1, 1],
    "LV": [1, 1, 0, 0]
}

f_pieces = ["FC", "FD", "FB", "FE"]
b_pieces = ["BC", "BD", "BB", "BE"]
v_pieces = ["VC", "VD", "VB", "VE"]
l_pieces = ["LH", "LV"]

class PipeManiaState:
    state_id = 0

    def __init__(self, board, changed_coords=None, changed_piece=None, heuristic=None, piece_index=0):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1
        self.changed_piece = changed_piece
        self.changed_coords = changed_coords
        self.piece_index = piece_index
        self.heuristic = heuristic

    
    def calculate_initial_heuristic(self):
        heuristic = 0
        dim = self.board.dimension

        for i in range(dim):
            for j in range(dim):
                piece = self.board.get_value(i, j)
                above = self.board.adjacent_vertical_values(i, j)[0]
                below = self.board.adjacent_vertical_values(i, j)[1]
                left = self.board.adjacent_horizontal_values(i, j)[0]
                right = self.board.adjacent_horizontal_values(i, j)[1]

                heuristic += self.calculate_piece_heuristic(piece, above, below, left, right)
        return heuristic

    def calculate_piece_heuristic(self, piece, above, below, left, right):
        piece_heuristic = 0

        if (above is None and pieces_dict[piece][0] == 1) or (above is not None and (pieces_dict[piece][0] == 1 and pieces_dict[above][1] == 0)):
            piece_heuristic += 1
        if (below is None and pieces_dict[piece][1] == 1) or (below is not None and (pieces_dict[piece][1] == 1 and pieces_dict[below][0] == 0)):
            piece_heuristic += 1
        if (left is None and pieces_dict[piece][2] == 1) or (left is not None and (pieces_dict[piece][2] == 1 and pieces_dict[left][3] == 0)):
            piece_heuristic += 1
        if (right is None and pieces_dict[piece][3] == 1) or (right is not None and (pieces_dict[piece][3] == 1 and pieces_dict[right][2] == 0)):
            piece_heuristic += 1

        return piece_heuristic


    def __lt__(self, other):
        return self.id < other.id

    def manual_copy(self):
        new_board = self.board.manual_copy()
        return PipeManiaState(new_board, self.changed_coords, self.changed_piece, self.heuristic, self.piece_index)



class Board:
    """Representação interna de um tabuleiro de PipeMania."""

    def __init__(self, grid: str, dim: int):
        self.grid = grid
        self.dimension = dim

    def get_value(self, row: int, col: int) -> str:
        index = (row * self.dimension + col) * 2
        return self.grid[index:index+2]

    def set_value(self, row: int, col: int, value: str):
        index = (row * self.dimension + col) * 2
        self.grid = self.grid[:index] + value + self.grid[index+2:]

    def print(self) -> str:
        result = ""
        for i in range(self.dimension):
            for j in range(self.dimension):
                if j != self.dimension-1:
                    result += self.get_value(i, j) + "\t"
                else:
                    result += self.get_value(i, j) 
            if i != self.dimension-1:
                result += "\n"
        return result

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        if row > 0:
            above = self.get_value(row-1, col)
        else:
            above = None
        if row < self.dimension - 1:
            below = self.get_value(row+1, col)  
        else:
            below = None

        return (above, below)

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        if col > 0:
            left = self.get_value(row, col-1)
        else: 
            left = None
        if col < self.dimension - 1:
            right = self.get_value(row, col+1)
        else: 
            right = None

        return (left, right)


    def manual_copy(self):
        return Board(self.grid, self.dimension)

    @staticmethod
    def parse_instance():
        """Lê o teste do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board."""
        grid = ""
        dim = 0
        for line in sys.stdin:
            dim += 1
            row = "".join(line.split())
            grid += row
        return Board(grid, dim)


class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board
        self.initial = PipeManiaState(board)
        self.pieces_queue = []
        

    def pre_processing(self):
        dim = self.board.dimension
        board = self.initial.board

        # Top Left Corner
        top_left = board.get_value(0, 0)
        if top_left in v_pieces:
            if top_left!="VB":
                board.set_value(0, 0, "VB")
        elif top_left in f_pieces:
            adj_v_pieces = board.adjacent_vertical_values(0, 0)
            adj_h_pieces = board.adjacent_horizontal_values(0, 0)

            if (adj_h_pieces[1] in (l_pieces or b_pieces)) or adj_v_pieces[1] in f_pieces:
                board.set_value(0, 0, "FD")
            elif (adj_v_pieces[1] in (l_pieces or b_pieces)) or adj_h_pieces[1] in f_pieces:
                board.set_value(0, 0, "FB")
            else:
                self.pieces_queue.append((0,0))
        else:
            self.pieces_queue.append((0,0))

        # Top Right Corner
        top_right = board.get_value(0, dim-1)
        if top_right in v_pieces:
            if top_right!="VE":
                board.set_value(0, dim-1, "VE")
        elif top_right in f_pieces:
            adj_v_pieces = board.adjacent_vertical_values(0, dim-1)
            adj_h_pieces = board.adjacent_horizontal_values(0, dim-1)

            if (adj_h_pieces[0] in (l_pieces or b_pieces)) or adj_v_pieces[1] in f_pieces:
                board.set_value(0, dim-1, "FE")
            elif (adj_v_pieces[1] in (l_pieces or b_pieces)) or adj_h_pieces[0] in f_pieces:
                board.set_value(0, dim-1, "FB")
            else:
                self.pieces_queue.append((0,dim-1))
        else:
            self.pieces_queue.append((0,dim-1))

        # Bottom Left Corner
        bottom_left = board.get_value(dim-1, 0)
        if  bottom_left in v_pieces:
            if bottom_left!="VD":
                board.set_value(dim-1, 0,"VD")
        elif bottom_left in f_pieces:
            adj_v_pieces = board.adjacent_vertical_values(dim-1, 0)
            adj_h_pieces = board.adjacent_horizontal_values(dim-1, 0)

            if (adj_h_pieces[1] in (l_pieces or b_pieces)) or adj_v_pieces[0] in f_pieces:
                board.set_value(dim-1, 0, "FD")
            elif (adj_v_pieces[0] in (l_pieces or b_pieces)) or adj_h_pieces[1] in f_pieces: 
                board.set_value(dim-1, 0, "FC")
            else:
                self.pieces_queue.append((dim-1,0))
        else:
            self.pieces_queue.append((dim-1,0))

        # Bottom Right Corner        
        bottom_right = board.get_value(dim-1, dim-1)
        if  bottom_right in v_pieces:
            if bottom_right!="VC":
                board.set_value(dim-1, dim-1,"VC")
        elif bottom_right in f_pieces:
            adj_v_pieces = board.adjacent_vertical_values(dim-1, dim-1)
            adj_h_pieces = board.adjacent_horizontal_values(dim-1, dim-1)

            if (adj_h_pieces[0] in (l_pieces or b_pieces)) or adj_v_pieces[0] in f_pieces:
                board.set_value(dim-1, dim-1, "FE")
            elif (adj_v_pieces[0] in (l_pieces or b_pieces)) or adj_h_pieces[0] in f_pieces: 
                board.set_value(dim-1, dim-1, "FC")
            else:
                self.pieces_queue.append((dim-1,dim-1))
        else:
            self.pieces_queue.append((dim-1,dim-1))


        for i in range(dim):
            for j in range(dim):
                if (i,j) not in [(0, 0), (0, dim-1), (dim-1, 0), (dim-1, dim-1)]:
                    piece = board.get_value(i, j)

                    # Upper and Bottom Edges
                    if i==dim-1 or i==0:
                        if piece in l_pieces: 
                            if piece !="LH":
                                board.set_value(i, j,"LH")
    
                        elif i==0 and piece in b_pieces:
                            if piece !="BB":
                                board.set_value(i, j,"BB")

                        elif i==dim-1 and piece in b_pieces:
                            if piece !="BC":
                                board.set_value(i, j,"BC")
                        else:
                            self.pieces_queue.append((i,j))

                    # Left and Right Edges
                    elif j==dim-1 or j==0:
                        if piece in l_pieces:
                            if piece !="LV":
                                board.set_value(i, j,"LV")

                        elif j==0 and piece in b_pieces:
                            if piece !="BD":
                                board.set_value(i, j,"BD")

                        elif j==dim-1 and piece in b_pieces:
                            if piece !="BE":
                                board.set_value(i, j,"BE")
                        else:
                            self.pieces_queue.append((i,j))
                    else:
                        self.pieces_queue.append((i,j))
        
        self.pieces_queue.sort()
        self.initial.heuristic = self.initial.calculate_initial_heuristic()

        """print("After processing:")
        print()
        print(self.initial.board.print())
        print(self.pieces_queue)"""
        return self

    def valid_rotation(self, state, coord, piece): 
        above = state.board.adjacent_vertical_values(coord[0], coord[1])[0]
        below = state.board.adjacent_vertical_values(coord[0], coord[1])[1]

        left = state.board.adjacent_horizontal_values(coord[0], coord[1])[0]
        right = state.board.adjacent_horizontal_values(coord[0], coord[1])[1]
        dim = self.board.dimension

        # checks piece above
        if above != None and pieces_dict[piece][0] != pieces_dict[above][1]:
            return False
        elif above == None and pieces_dict[piece][0] == 1:
            return False
        
        # checks piece to the left
        if left != None and pieces_dict[piece][2] != pieces_dict[left][3]:
            return False
        elif left == None and pieces_dict[piece][2] == 1:
            return False
        
        # checks piece below if it has been solved
        if coord[0]!=dim-1:
            if (coord[0] + 1, coord[1]) not in self.pieces_queue:
                if below != None and pieces_dict[piece][1] != pieces_dict[below][0]:
                    return False
                elif below == None and pieces_dict[piece][1] == 1:
                    return False

        # checks piece to the right if it has been solved
        if coord[1]!=dim-1:
            if (coord[0], coord[1]+1) not in self.pieces_queue:
                if right != None and pieces_dict[piece][3] != pieces_dict[right][2]:
                    return False
                elif right == None and pieces_dict[piece][3] == 1:
                    return False

        return True

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        dim = self.board.dimension

        actions = []
        coord = ()
        if state.piece_index >= len(self.pieces_queue):
            return actions
        coord = self.pieces_queue[state.piece_index]
        state.piece_index += 1
        piece = state.board.get_value(coord[0], coord[1])
        horizontal_values = self.board.adjacent_horizontal_values(coord[0],coord[1])
        vertical_values = self.board.adjacent_vertical_values(coord[0],coord[1])
        # skips known invalid rotations in certains coords
        for i in range(4):
            if piece in f_pieces:
                new_piece = f_pieces[(f_pieces.index(piece) - i) % 4]
                if (horizontal_values[0] == None or horizontal_values[0] in f_pieces) and new_piece == "FE":
                    continue
                if (horizontal_values[1] == None or horizontal_values[1] in f_pieces) and new_piece == "FD":
                    continue
                if (vertical_values[0] == None or vertical_values[0] in f_pieces) and new_piece == "FC":
                    continue
                if (vertical_values[1] == None or vertical_values[1] in f_pieces) and new_piece == "FB":
                    continue
            elif piece in b_pieces:
                new_piece = b_pieces[(b_pieces.index(piece) - i) % 4]
                if coord[0] == 0 and new_piece != "BB":
                    continue
                elif coord[0] == dim-1 and new_piece != "BC":
                    continue
                elif coord[1] == 0 and new_piece != "BD":
                    continue
                elif coord[1] == dim-1 and new_piece != "BE":
                    continue

            elif piece in v_pieces:
                new_piece = v_pieces[(v_pieces.index(piece) - i) % 4]
                if coord[0] == 0 and new_piece in ["VC", "VD"]:
                    continue
                elif coord[0] == dim-1 and new_piece in ["VB", "VE"]:
                    continue
                elif coord[1] == 0 and new_piece in ["VC", "VE"]:
                    continue
                elif coord[1] == dim-1 and new_piece in ["VB", "VD"]:
                    continue
            
            elif piece in l_pieces:
                if i<2:
                    new_piece = l_pieces[(l_pieces.index(piece) - i) % 2]
                else: 
                    continue
            if self.valid_rotation(state, coord, new_piece):
                actions.append((coord[0], coord[1], i))
        #print(actions)
        return actions

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        new_state = state.manual_copy() 

        piece = state.board.get_value(action[0], action[1])

        new_state.changed_coords = (action[0], action[1])
        new_state.changed_piece = piece

        # roda em sentido contrario ao do relogio (action[2] indica quando vezes gira 90º)
        if piece in f_pieces:
            new_state.board.set_value(action[0], action[1], f_pieces[(f_pieces.index(piece) - action[2]) % 4])
        elif piece in b_pieces:
            new_state.board.set_value(action[0], action[1], b_pieces[(b_pieces.index(piece) - action[2]) % 4])
        elif piece in v_pieces:
            new_state.board.set_value(action[0], action[1], v_pieces[(v_pieces.index(piece) - action[2]) % 4])
        elif piece in l_pieces:
            new_state.board.set_value(action[0], action[1], l_pieces[(l_pieces.index(piece) - action[2]) % 2])
        """print("RESUlT:")
        print(new_state.board.print())"""
        return new_state


    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        dim = self.board.dimension
        block = []
        check_connections = [(0,0)]
        checked_connections = []
        if state.heuristic>0:
            #print("GOAL:", state.heuristic)
            return False
        
        #percorre o tabuleiro pelas ligações como um virus e no final vê se o numero de peças "infetadas" é igual ao tamanho da grid
        while len(check_connections) > 0:
            coords = check_connections.pop(0)
            piece_shape = pieces_dict[state.board.get_value(coords[0], coords[1])]
            above = (coords[0]-1, coords[1])
            below = (coords[0]+1, coords[1])
            left = (coords[0], coords[1]-1)
            right = (coords[0], coords[1]+1)
            if (piece_shape[0]==1 and above not in checked_connections and above not in check_connections):
                check_connections.append(above)
            if (piece_shape[1]==1 and below not in checked_connections and below not in check_connections):
                check_connections.append(below)
            if (piece_shape[2]==1 and left not in checked_connections and left not in check_connections):
                check_connections.append(left)
            if (piece_shape[3]==1 and right not in checked_connections and right not in check_connections):
                check_connections.append(right)
            checked_connections.append(coords)
            block.append(coords)

        if len(block)!= dim*dim:
            return False
        return True

    def h(self, node: Node):
        """Heuristic function used for A* search."""
        if node.state.changed_coords is not None:
            i, j = node.state.changed_coords
            new_piece = node.state.board.get_value(i, j)
            changed_piece = node.state.changed_piece
            above = node.state.board.adjacent_vertical_values(i, j)[0]
            below = node.state.board.adjacent_vertical_values(i, j)[1]
            left = node.state.board.adjacent_horizontal_values(i, j)[0]
            right = node.state.board.adjacent_horizontal_values(i, j)[1]

            node.state.heuristic -= node.state.calculate_piece_heuristic(changed_piece, above, below, left, right)
            node.state.heuristic += node.state.calculate_piece_heuristic(new_piece, above, below, left, right)

            # Below
            if below is not None:
                below_below = node.state.board.adjacent_vertical_values(i + 1, j)[1]
                below_left = node.state.board.adjacent_horizontal_values(i + 1, j)[0]
                below_right = node.state.board.adjacent_horizontal_values(i + 1, j)[1]
                node.state.heuristic -= node.state.calculate_piece_heuristic(below, changed_piece, below_below, below_left, below_right)
                node.state.heuristic += node.state.calculate_piece_heuristic(below, new_piece, below_below, below_left, below_right)

            # Above
            if above is not None:
                above_above = node.state.board.adjacent_vertical_values(i - 1, j)[0]
                above_left = node.state.board.adjacent_horizontal_values(i - 1, j)[0]
                above_right = node.state.board.adjacent_horizontal_values(i - 1, j)[1]
                node.state.heuristic -= node.state.calculate_piece_heuristic(above, above_above, changed_piece, above_left, above_right)
                node.state.heuristic += node.state.calculate_piece_heuristic(above, above_above, new_piece, above_left, above_right)

            # Left
            if left is not None:
                left_above = node.state.board.adjacent_vertical_values(i, j - 1)[0]
                left_below = node.state.board.adjacent_vertical_values(i, j - 1)[1]
                left_left = node.state.board.adjacent_horizontal_values(i, j - 1)[0]
                node.state.heuristic -= node.state.calculate_piece_heuristic(left, left_above, left_below, left_left, changed_piece)
                node.state.heuristic += node.state.calculate_piece_heuristic(left, left_above, left_below, left_left, new_piece)

            # Right
            if right is not None:
                right_above = node.state.board.adjacent_vertical_values(i, j + 1)[0]
                right_below = node.state.board.adjacent_vertical_values(i, j + 1)[1]
                right_right = node.state.board.adjacent_horizontal_values(i, j + 1)[1]
                node.state.heuristic -= node.state.calculate_piece_heuristic(right, right_above, right_below, changed_piece, right_right)
                node.state.heuristic += node.state.calculate_piece_heuristic(right, right_above, right_below, new_piece, right_right)

        return node.state.heuristic


if __name__ == "__main__":
    board = Board.parse_instance()
    problem = PipeMania(board)
    problem = problem.pre_processing()

    solution_node = astar_search(problem, problem.h)
    print(solution_node.state.board.print())
    exit(0)