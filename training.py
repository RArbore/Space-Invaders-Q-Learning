from random import randrange as rand
import multiprocessing
import numpy as np
from multiprocessing import shared_memory
import torch
import math
import random
from torchvision import transforms
import time
import copy
import os
import sys

episodes = 25000

load_previous = False

evaluation = False

make_trial = True

initial_greedy_threshold = 0.05
greedy_decay = 1

if evaluation:
    initial_greedy_threshold = 0

gamma = 0.9

frames_per_state = 3

lr = 0.0001

b1 = 0.5
b2 = 0.999

batch_size = 250

folder = ""

class QNetwork(torch.nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()
        # self.conv = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 5, 3, 1, 1),
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.ReLU(True),
        #     torch.nn.Conv2d(5, 7, 3, 1, 1),
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.ReLU(True),
        #     torch.nn.Conv2d(7, 9, 3, 1, 1),
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.ReLU(True),
        #     torch.nn.Conv2d(9, 11, 3, 1, 1),
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.ReLU(True),
        # )
        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(13, 10),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(10, 5),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(5, 1),
        )
        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(13, 10),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(10, 5),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(5, 3),
        )

    def forward(self, input, xpos, bpos):
        # out = []
        # for i in input:
        #     out.append(torch.mean(i.view(1, 3, 600, 800).float()/256.0, dim=1, keepdim=True))
        # out = torch.cat(out, dim=1)
        # out = out[:, :, 200:600, :]
        # features = self.conv(out)
        # features = features.view(out.size(0), -1)

        features = xpos.view(1, 1).float()
        features = torch.cat((features, bpos.view(1, 12).float()), dim=1)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

def save_image(tensor, filename):
    ndarr = tensor.int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)

def training_stuff(shm_screen_name, shm_stats_name, shm_controls_name, shm_gameover_name, shm_player_input_name):
    current_milli_time = lambda: int(round(time.time() * 1000))

    if make_trial:
        files = os.listdir(".")
        folder_number = 1

        m = [int(f[5:]) for f in files if len(f) > 5 and f[0:5] == "trial"]
        if len(m) > 0:
            folder = "trial" + str(max(m) + 1)
            folder_number = max(m) + 1
        else:
            folder = "trial1"
        os.mkdir(folder)

        print("Created session folder " + folder)

        f = open(folder + "/scores.txt", "a")

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    cpu = torch.device("cpu")

    model = QNetwork().to(device)

    if load_previous and folder_number > 1:
        print("Loading from trial "+str(folder_number - 1))
        model.load_state_dict(torch.load("trial"+str(folder_number - 1)+"/trained.pt"))

    if evaluation:
        model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

    # action_matrix = [
    #     [0, 0, 0],
    #     [0, 0, 1],
    #     [1, 0, 0],
    #     [1, 0, 1],
    #     [0, 1, 0],
    #     [0, 1, 1]
    # ]

    action_matrix = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]

    replay_buffer = []

    state = None
    action = None
    reward = None
    last_reward = None
    state_prime = None
    calculated_utility = None

    current_milli_time = lambda: int(round(time.time() * 1000))

    screen_attach = shared_memory.SharedMemory(name=shm_screen_name)
    stats_attach = shared_memory.SharedMemory(name=shm_stats_name)
    controls_attach = shared_memory.SharedMemory(name=shm_controls_name)
    player_input_attach = shared_memory.SharedMemory(name=shm_player_input_name)
    gameover_attach = shared_memory.SharedMemory(name=shm_gameover_name)

    episode = 0

    average_q = []
    average_r = []

    last_lives = 0

    while episode < episodes:
        state = []

        # for i in range(frames_per_state):
        #     raw = screen_attach.buf.tobytes()
        #     image_tensor = torch.tensor(np.frombuffer(raw, dtype=np.uint8))[0:1920000].view(600, 800, 4).permute(2, 0, 1)[0:3]
        #     #image_tensor = torch.nn.functional.interpolate(image_tensor.float().view(1,3, 600, 800), size=(300, 400))
        #     state.append(image_tensor.view(3, 600, 800))

        raw = stats_attach.buf.tobytes()
        l = list(raw[0:17])
        stats_tensor = torch.zeros(2)
        stats_tensor[0] = l[0]+256*l[1]+256*256*l[2]
        stats_tensor[1] = l[3] - 1
        xpos = l[4]
        bpos = []
        for i in range(5, 17):
            bpos.append(l[i])

        score = stats_tensor[1]-last_lives
        # if not stats_tensor[1] == 0:
        #     print(stats_tensor[1])
        average_r.append(score)
        last_lives = stats_tensor[1].item()

        model_input = []

        for i in state:
            model_input.append(i.to(device))

        utility_values = model(model_input, torch.tensor(xpos).to(device), torch.tensor(bpos).to(device)).view(-1)
        average_q.append(torch.max(utility_values).item())
        del model_input
        calculated_action = torch.argmax(utility_values).cpu().item()
        epsilon = random.random()
        if epsilon < initial_greedy_threshold*(greedy_decay**episode):
            calculated_action = random.randint(0, 3-1)

        controls_attach.buf[0] = action_matrix[calculated_action][0]
        controls_attach.buf[1] = action_matrix[calculated_action][1]
        controls_attach.buf[2] = action_matrix[calculated_action][2]

        # controls_attach.buf[0] = player_input_attach.buf[0]
        # controls_attach.buf[1] = player_input_attach.buf[1]
        # controls_attach.buf[2] = player_input_attach.buf[2]

        action = calculated_action

        #if gameover_attach.buf[0] == 0:
        replay_buffer.append((state, action, score, stats_tensor[0], xpos, bpos))
        if len(replay_buffer) > 5000:
            gameover_attach.buf[0] = 1

            #time.sleep(0.1)

        if gameover_attach.buf[0] == 1 and len(replay_buffer) > 2 and score < 0:
            if not evaluation:
                max_score = 0
                replays_by_reward = []

                pred_model = copy.deepcopy(model).to(device)

                #print(len(replay_buffer))

                for r in range(1, len(replay_buffer)):
                    replays_by_reward.append(abs(replay_buffer[r][2] - replay_buffer[r - 1][2])+10)

                if len(replays_by_reward) > 0:
                    episode += 1

                    for r in range(1, len(replay_buffer)-1):
                        max_score = max(max_score, replay_buffer[r][3].item())
                    
                    replays_by_reward = np.array(replays_by_reward)

                    norm = np.linalg.norm(replays_by_reward, ord=1)

                    samples = np.random.choice(range(1, len(replays_by_reward)+1), batch_size)#, p=replays_by_reward/norm)

                    #samples = [np.where(replays_by_reward == np.amin(replays_by_reward))[0][0]]

                    avg_loss = 0.0
                    for sample in samples:
                        loss = torch.zeros(1).to(device)
                        if replay_buffer[sample][2] - replay_buffer[sample-1][2] > 0:
                            continue
                        # for i in range(frames_per_state):
                        #     replay_buffer[sample][0][i] = replay_buffer[sample][0][i].to(device)
                        #     replay_buffer[sample+1][0][i] = replay_buffer[sample+1][0][i].to(device)
                        target = (replay_buffer[sample][2] - replay_buffer[sample-1][2]).to(device)
                        if not (replay_buffer[sample][2] == -1):
                            target += gamma*torch.max(pred_model(replay_buffer[sample+1][0], torch.tensor(replay_buffer[sample][4]).to(device), torch.tensor(replay_buffer[sample][5]).to(device))).to(device)
                        loss += ((target - model(replay_buffer[sample][0], torch.tensor(replay_buffer[sample][4]).to(device), torch.tensor(replay_buffer[sample][5]).to(device))[0, replay_buffer[sample][1]])**2).float().to(device)
                        # for i in range(frames_per_state):
                        #     replay_buffer[sample][0][i] = replay_buffer[sample][0][i].to(cpu)
                        #     replay_buffer[sample+1][0][i] = replay_buffer[sample+1][0][i].to(cpu)

                        loss.backward()
                        avg_loss += loss.item()
                        opt.step()
                        opt.zero_grad()

                    avg_loss /= float(len(samples))

                    q = np.mean(average_q)
                    r = np.mean(average_r)

                    average_q = []
                    average_r = []

                    last_lives = 0

                    print("Episode "+str(episode)+"   Score: "+str(max_score)+"   Reward: "+str(r)+"   Loss: "+str(avg_loss)+"   Q: "+str(q))

                    if make_trial:
                        f.write(str(episode)+" "+str(max_score)+" "+str(r)+" "+str(avg_loss)+" "+str(q)+"\n")

                replay_buffer = []

            gameover_attach.buf[0] = 0
            time.sleep(0.25)
        elif gameover_attach.buf[0] == 1 and score < 0:
            gameover_attach.buf[0] = 0

    if make_trial:
        f.close()
        torch.save(model.state_dict(), folder + "/trained.pt")