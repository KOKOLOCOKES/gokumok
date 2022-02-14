#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import time
from tqdm import tqdm 
import copy

#----------------------------------------------------------------------------------------------------------------------------
BRIGHT_RED = '\033[101m'
BRIGHT_BLUE = '\033[94m'
BRIGHT_GREEN = '\033[106m'
BRIGHT_BLACK = '\033[90m'
BRIGHT_END = '\033[0m'
MAGENTA = '\033[35m'
#----------------------------------------------------------------------------------------------------------------------------


class Environment():
    pos = None  #마지막으로 놓은 돌의 위치 저장

    def get_pos(self): #class 변수를 반환
        return self.pos

    def __init__(self):
        self.board = np.zeros(81) #돌을 놓는 보드
        self.influence_board = np.zeros(81) #영향력을 계산하는 보드
        self.done = False #게임 종료 판정
        self.reward = 0 #승리한 플레이어의 정보를 저장
        self.winner = 0 
        self.print = False
        self.att_check = False #처음으로 영향력이 생기는 것을 확인(초반 MC 범위 제한에 사용)
        
    def add_influence(self, pos, player):
        #detector[0]: 돌 사이 공백이 있는 경우 탐색
        #detector[1]: 놓은 돌 주변 8칸 탐색
        detector = [[pos-20, pos-18, pos-16, pos-2, pos+2, pos+16, pos+18, pos+20],                   [pos-10, pos-9, pos-8, pos-1, pos+1, pos+8, pos+9, pos+10]]
        
        #놓은 돌과 같은 종류(player)의 돌의 좌표를 저장
        save_position=[[],[]]
        
        for i in range(2):
            for j in detector[i]:
                if j >= 0 and j <=80 and self.board[j] == player:
                    save_position[i].append(j)
                    self.att_check = True
                
                
        #돌을 놓은 자리에 영향력을 더함
        self.influence_board[pos] += 8*player
        
        
        if len(save_position) != 0:
            
            #save_position 2차원 배열 계산
            for u in range(2):
                for k in save_position[u]:
                    
                    #현재좌표와 탐색된 좌표 사이의 차를 저장
                    gap = pos - k
                    
                    if u == 0:
                        #두 좌표 사이의 중간 값을 기준으로 함
                        gap = int(gap*0.5)
                        m_pos = pos - gap
                        influence = [m_pos - gap,  m_pos - gap*2, m_pos - gap*3, m_pos + gap, m_pos + gap*2, m_pos + gap*3]
                        
                    else:
                        #현재 놓은 좌표를 기준으로 함
                        influence = [pos - gap*2, pos - gap*3,pos - gap*4, pos + gap, pos + gap*2,pos + gap*3]
                        
                    #영향력 값을 줄 좌표를 저장
                    pos_list=[]

                    # 몇 번째에 벽(cliff)이 있는지 확인
                    #세 칸까지만 탐색하기 때문에 네 번째에서의 벽의 유무는 의미가 없다
                    cliff = 4 
                    
                    #왼쪽, 오른쪽 값을 같게 주기 위해서 사용
                    score_change = False 
                    
                    #오른쪽, 왼쪽 두번 탐색
                    for i in range(2):
                        
                        Left_Right = []
                        
                        if i == 0:
                            Left_Right = influence[0:3]
                            
                        else:
                            Left_Right = influence[3:6]

                        for h in range(1,4):
                            
                            #돌에서 가까운 순서대로 하나씩 판별
                            c_check = Left_Right[h-1]

                            ##########################################################
                            #벽을 탐색하고, 벽을 발견하면 영향력이 더 뻗어나가지 못하게 함
                            
                            #보드 영역을 넘기는 경우
                            if c_check < 0  or c_check > 80:
                                
                                #놓은 돌과 가장 가까운 벽의 위치를 저장함
                                if cliff > h:
                                    cliff = h
                                break
                                
                            #줄이 바뀌는 경우
                            #줄의 양 끝점을 지나는 경우를 탐색하지만 양 끝점에서 세로로 탐색하는 경우는 제외
                            elif (c_check % 9 == 0 or (c_check - 8) % 9 == 0) and  gap != 9 and gap != -9:
                                
                                #놓은 돌의 좌표가 끝점일 경우는 영향력을 뻗지 않음
                                if not ((k or pos) % 9 == 0 and (c_check - 8) % 9 == 0) or                                (((k or pos)-8) % 9 == 0 or c_check % 9 == 0):
                                    
                                    if cliff > h+1 and h != 3:
                                        cliff = h+1
                                    
                                    #한 줄의 양 끝점을 탐색하는 경우(줄이 바뀌기 직전)
                                    #줄을 바꾸는 순간을 벽으로 하기 때문에, 양 끝점은 벽으로 판단하지 않음
                                    pos_list.append(c_check)
                                    
                                else:
                                    if cliff > h and h != 3:
                                        cliff = h
                                break
                            
                            #다른 플레이어 돌이 있는 경우
                            elif self.board[c_check] == -player:
                                
                                if cliff > h:
                                    cliff = h
                                    
                                break
                                
                            #벽에 걸리지 않으면 좌표를 저장
                            pos_list.append(c_check)
                            ##########################################################
                            
                    #좌우를 나누어 내림차순으로 정렬
                    pos_list.sort(reverse=True, key = lambda x: x <= pos)
                    
                    if u == 0:
                        #두 돌 사이의 칸에 가치를 줌
                        #영향력의 부호가 이미 정해진 경우, 그 부호를 따라감
                        if self.influence_board[m_pos] != 0:
                            #(0.6+0.1*cliff) : 벽의 위치에 따라 영향력 비율을 감소시킴
                            self.influence_board[m_pos] += 4*(0.6+0.1*cliff)*np.sign(self.influence_board[m_pos])
                        else:
                            self.influence_board[m_pos] += 4*(0.6+0.1*cliff)*player
                            
                        #중간값을 기준으로 0.5배 감소된 영향력으로 뻗어나감
                        score = 2
                        
                    else:
                        score = 4
                        
                    for i in pos_list:
                        if pos < i and score_change == False:
                            if u == 0:
                                score = 2
                            else:
                                score = 4
                            score_change = True

                        if self.influence_board[i] != 0:
                            self.influence_board[i] += score*np.sign(self.influence_board[i])*(0.6+0.1*cliff)
                        else:
                            self.influence_board[i] += score*player*(0.6+0.1*cliff)
                            
                        #영향력 좌표에 같은 플레이어의 돌이 있을 경우, 영향력을 감소시키지 않음
                        if self.board[i] != player:
                            score *= 0.5
                    
#__________________________________________________________________________________________________________________
                   
    def move(self, p1, p2, player):
        
        #번갈아가며 돌을 놓게 함
        if player == 1:
            self.pos = p1.select_action(env, player)
        else:
            self.pos = p2.select_action(env, player)
        
        #보드에 돌을 놓음
        self.board[self.pos] = player
        
        #놓은 돌을 기준으로 영향력을 계산함
        self.add_influence(self.pos, player)
        
        if self.print:
            print(player)
            self.print_board()
            self.print_influence_board()
        
        #게임이 끝났는지 확인
        self.end_check(player)
        
        return self.reward, self.done
     
    def get_action(self):
        observation = []
        for i in range(81):
            #영향력이 있으면서 돌을 놓을 수 있는 위치를 찾고 저장
            if self.influence_board[i] != 0 and self.board[i] == 0:
                observation.append(i)
        
        #절대값이 가장 큰 영향력을 내림차순으로 정렬함
        observation.sort(reverse = True, key = lambda x:abs(self.influence_board[x]))
        
        #적당한 수의 영향력이 생기기 전까지는 Default값을 사용해 계산
        if len(observation) <= 13:
            observation = observation[0:3]
        else:
            #총 영향력 개수에서 일정%를 추출
            observation=observation[0:int(len(observation)*0.23)]
                
        return observation
    

    def all_action(self):
        observation = []
        for i in range(81):
            #영향력이 없더라도 놓을 수 있는 좌표를 저장
            if self.board[i] == 0:
                observation.append(i)
        return observation


    def end_check(self, player):
        end_condition = (((4,12,20,28,36),(5,13,21,29,37),(13,21,29,37,45),(6,14,22,30,38),         (14,22,30,38,46),(22,30,38,46,54),(7,15,23,31,39),(15,23,31,39,47),(23,31,39,47,55),           (31,39,47,55,63),(8,16,24,32,40),(16,24,32,40,48),(24,32,40,48,56),(32,40,48,56,64),           (40,48,56,64,72),(17,25,33,41,49),(25,33,41,49,57),(33,41,49,57,65),(41,49,57,65,73),           (26,34,42,50,58),(34,42,50,58,66),(42,50,58,66,74),(35,43,51,59,67),(43,51,59,67,75), (44,52,60,68,76)),            ((4,14,24,34,44),(3,13,23,33,43),(13,23,33,43,53),(2,12,22,32,42),            (12,22,32,42,52),(22,32,42,52,62),(1,11,21,31,41),(11,21,31,41,51),(21,31,41,51,61),           (31,41,51,61,71),(0,10,20,30,40),(10,20,30,40,50),(20,30,40,50,60),(30,40,50,60,70),           (40,50,60,70,80),(9,19,29,39,49),(19,29,39,49,59),(29,39,49,59,69),(39,49,59,69,79),           (18,28,38,48,58),(28,38,48,58,68),(38,48,58,68,78),(27,37,47,57,67),(37,47,57,67,77),(36,46,56,66,76)),            ((0,1,2,3,4),(1,2,3,4,5),(2,3,4,5,6),(3,4,5,6,7),(4,5,6,7,8),           (9,10,11,12,13),(10,11,12,13,14),(11,12,13,14,15),(12,13,14,15,16),(13,14,15,16,17),           (18,19,20,21,22),(19,20,21,22,23),(20,21,22,23,24),(21,22,23,24,25),(22,23,24,25,26),           (27,28,29,30,31),(28,29,30,31,32),(29,30,31,32,33),(30,31,32,33,34),(31,32,33,34,35),           (36,37,38,39,40),(37,38,39,40,41),(38,39,40,41,42),(39,40,41,42,43),(40,41,42,43,44),           (45,46,47,48,49),(46,47,48,49,50),(47,48,49,50,51),(48,49,50,51,52),(49,50,51,52,53),           (54,55,56,57,58),(55,56,57,58,59),(56,57,58,59,60),(57,58,59,60,61),(58,59,60,61,62),           (63,64,65,66,67),(64,65,66,67,68),(65,66,67,68,69),(66,67,68,69,70),(67,68,69,70,71),           (72,73,74,75,76),(73,74,75,76,77),(74,75,76,77,78),(75,76,77,78,79),(76,77,78,79,80)),           ((0,9,18,27,36),(1,10,19,28,37),(2,11,20,29,38),(3,12,21,30,39),(4,13,22,31,40),(5,14,23,32,41),           (6,15,24,33,42),(7,16,25,34,43),(8,17,26,35,44),(9,18,27,36,45),(10,19,28,37,46),(11,20,29,38,47),           (12,21,30,39,48),(13,22,31,40,49),(14,23,32,41,50),(15,24,33,42,51),(16,25,34,43,52),(17,26,35,44,53),           (18,27,36,45,54),(19,28,37,46,55),(20,29,38,47,56),(21,30,39,48,57),(22,31,40,49,58),(23,32,41,50,59),           (24,33,42,51,60),(25,34,43,52,61),(26,35,44,53,62),(27,36,45,54,63),(28,37,46,55,64),(29,38,47,56,65),           (30,39,48,57,66),(31,40,49,58,67),(32,41,50,59,68),(33,42,51,60,69),(34,43,52,61,70),(35,44,53,62,71),           (36,45,54,63,72),(37,46,55,64,73),(38,47,56,65,74),(39,48,57,66,75),(40,49,58,67,76),(41,50,59,68,77),           (42,51,60,69,78),(43,52,61,70,79),(44,53,62,71,80)))

        for line in end_condition:
            #6개가 연속일 경우 : 승리로 계산하지 않음
            six_check = 0
            for i in range(len(line)):
                if self.board[line[i][0]] == self.board[line[i][1]]                and self.board[line[i][1]] == self.board[line[i][2]]                and self.board[line[i][2]] == self.board[line[i][3]]                and self.board[line[i][3]] == self.board[line[i][4]]                and self.board[line[i][0]] != 0:
                    six_check += 1
            
            #가로/세로/대각선으로 나누어 볼 때, 승리조건이 두번 만족하면 6개가 이어진것으로 봄
            if six_check == 1:
                self.done = True
                self.reward = player
                return
            


        observation = self.all_action()
        
        #놓을 자리가 없을 경우 = 게임종료, 무승부
        if (len(observation)) == 0:
            self.done = True
            self.reward = 0
        return
    
    def print_board(self):
        print("+----+----+----+----+----+----+----+----+----+")
        for i in range(9):
            for j in range(9):
                if self.board[9*i+j] == 1:
                    print("|  O",end=" ")
                elif self.board[9*i+j] == -1:
                    print("|  X",end=" ")
                else:
                    print("|   ",end=" ")
            print("|")
            print("+----+----+----+----+----+----+----+----+----+")
            
    def print_influence_board(self):
        print("+-----+-----+-----+-----+-----+-----+-----+-----+-----+")
        for i in range(9):
            for j in range(9):
                if self.board[9*i+j] == -1:
                    print("|", BRIGHT_RED+"{:.2f}".format(self.influence_board[9*i+j])+BRIGHT_END,sep = '', end="")
                elif self.board[9*i+j] == 1:
                    print("|", BRIGHT_GREEN+"{:.2f}".format(self.influence_board[9*i+j])+BRIGHT_END,sep = '', end=" ")
                elif self.influence_board[9*i+j] > 0:
                    print("|", BRIGHT_BLUE+"{:.2f}".format(self.influence_board[9*i+j])+BRIGHT_END,sep = '', end=" ")
                elif self.influence_board[9*i+j] < 0:
                    print("|", MAGENTA+"{:.2f}".format(self.influence_board[9*i+j])+BRIGHT_END,sep = '', end="")
                else:
                    print("|", BRIGHT_BLACK+"{:.2f}".format(self.influence_board[9*i+j])+BRIGHT_END,sep = '', end=" ")
                    
            print("|")
            print("+-----+-----+-----+-----+-----+-----+-----+-----+-----+")
            
    def print_board_human(self):
        print("+----+----+----+----+----+----+----+----+----+")
        for i in range(9):
            for j in range(9):
                if self.board[9*i+j] == -1:
                    print("|", BRIGHT_RED+"{}".format(9*i+j)+BRIGHT_END,sep = ' ', end=" ")
                    if 9*i+j < 10:
                        print(" ", sep='', end='')
                elif self.board[9*i+j] == 1:
                    print("|", BRIGHT_GREEN+"{}".format(9*i+j)+BRIGHT_END,sep = ' ', end=" ")
                    if 9*i+j < 10:
                        print(" ", sep='', end='')
                elif self.board[9*i+j] == 0:
                    print("|", BRIGHT_BLACK+"{}".format(9*i+j)+BRIGHT_END,sep = ' ', end=" ")
                    if 9*i+j < 10:
                        print(" ", sep='', end='')
            
            print("|")
            print("+----+----+----+----+----+----+----+----+----+")
            
#----------------------------------------------------------------------------------------------------------------------------

class Monte_Carlo_player():
   
    def __init__(self):
        self.name = "MC player"
        self.num_playout = 75
        
        #첫 돌은 40에 놓게 한다(선공 기준)
        self.first_action = False

    def select_action(self, env, player):
               
        #영향력이 생기기 전까지는 놓을 수 있는 범위를 제한함
        if env.get_pos() ==  None or env.att_check != True:
            if self.first_action == False and env.board[40] == 0:
                return 40
            else:
                around = [30,31,32,39,40,41,48,49,50]
                while True:
                    action = np.random.randint(len(around))
                    if env.board[around[action]] == 0:
                        break
                return around[action]
        
        #놓을 수 있는 좌표를 받아옴
        available_action = env.get_action()
    
        #영향력이 없을 경우, 모든 좌표에서 계산을 함
        if len(available_action) == 0:
            available_action = env.all_action()
            
        #좌표의 수만큼의 크기를 가진 V라는 리스트를 만듦
        V = np.zeros(len(available_action))

        for i in range(len(available_action)):
            for j in range(self.num_playout):
                #계산 시간이 빠르기 때문에 랜덤 seed로 소수점자리를 가져옴
                np.random.seed(int((time.time()-int(time.time()))*1000000))
                temp_env = copy.deepcopy(env)
                self.playout(temp_env, available_action[i], player)
                
                #temp_env에서의 승패 결과에 따라 보상을 줌
                if player == temp_env.reward:
                    V[i] += 1
#                 elif -player == temp_env.reward:
#                     V[i] -= 1
#                 elif 0 == temp_env.reward:
#                     V[i] += 0.1
                    
        #보상값이 가장 큰 것을 선택
        action = np.argmax(V)
        return available_action[action]

    def playout(self, temp_env, action, player):
        temp_env.board[action] = player
        temp_env.add_influence(action, player)
        temp_env.end_check(player)
        

        if temp_env.done == True:
            return 
        else:
            player = -player
        
            available_action = temp_env.get_action()
            if len(available_action)==0:
                available_action = temp_env.all_action()

            action = np.random.randint(len(available_action))
            self.playout(temp_env, available_action[action], player)
                  
























class Human_player():
    
    def __init__(self):
        self.name = "Human player"
    
    def select_action(self, env, player):
        while True:
            # 가능한 행동을 조사한 후 표시
            possible_action = env.all_action()
            print("possible actions = {}".format(possible_action))
            
            env.print_board_human()
                        
            # 키보드로 가능한 행동을 입력 받음
            action = input("Select action(human) : ")
            action = int(action)
            
            # 입력받은 행동이 가능한 행동이면 반복문을 탈출
            if action in possible_action:
                return action
            # 아니면 행동 입력을 반복
            else:
                print("You selected wrong action")
        return

    
    
p1 = Monte_Carlo_player()

p2 = Human_player()

# 지정된 게임 수를 자동으로 두게 할 것인지 한게임씩 두게 할 것인지 결정
# auto = True : 지정된 판수(games)를 자동으로 진행 
# auto = False : 한판씩 진행

auto = False

# auto 모드의 게임수
games = 10

print("pl player : {}".format(p1.name))
print("p2 player : {}".format(p2.name))

# 각 플레이어의 승리 횟수를 저장
p1_score = 0
p2_score = 0
draw_score = 0


if auto: 
    # 자동 모드 실행
    for j in tqdm(range(games)):
        
        np.random.seed(j)
        env = Environment()
        
        for i in range(100000):
            # p1 과 p2가 번갈아 가면서 게임을 진행
            # p1(1) -> p2(-1) -> p1(1) -> p2(-1) ...
            reward, done = env.move(p1,p2,(-1)**i)
            # 게임 종료 체크
            if done == True:
                if reward == 1:
                    p1_score += 1
                elif reward == -1:
                    p2_score += 1
                else:
                    draw_score += 1
                    
                print("Final result")
                env.print_board()
                print("최대 개수: ", 81-len(env.all_action()))
                
                break

else:                
    # 한 게임씩 진행하는 수동 모드
    np.random.seed(int(time.time()))
    while True:
        
        env = Environment()
        env.print = False
        for i in range(100000):
            reward, done = env.move(p1,p2,(-1)**i)
#            env.print_board()
            env.print_influence_board()
            if done == True:
                if reward == 1:
                    print("winner is p1({})".format(p1.name))
                    p1_score += 1
                elif reward == -1:
                    print("winner is p2({})".format(p2.name))
                    p2_score += 1
                else:
                    print("draw")
                    draw_score += 1
                break
        
        # 최종 결과 출력        
        print("final result")
        env.print_board()

        # 한게임 더?최종 결과 출력 
        answer = input("More Game? (y/n)")

        if answer == 'n':
            break           

print("p1({}) = {} p2({}) = {} draw = {}".format(p1.name, p1_score,p2.name, p2_score,draw_score))


# In[ ]:

