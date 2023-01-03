#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 20:08:18 2022

@author: sijingyu
"""

import torch
import torch.nn.utils.prune as prune
from train import model_train
import sys



def one_shot_global_pruning(train_model, amount_ratio):
    ###  do the one shot global pruning  ###
    ###  we just do pruning of Conv and Lieanr layer which not include BN or other layers.
    parameters_to_prune = []
    for module in train_model.model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount_ratio)
    return train_model.model


def compute_module_sparsity(module):
    ###  compute the sparsity of each module, such as layer ###
    total = sum([param.nelement() for param in module.parameters()])
    num_zero = sum([torch.sum(buffer == 0).item() for buffer in module.buffers()])
    return num_zero, total


def compute_global_sparsity(net):
    ### compute the global sparsity by calling compute_module_sparsity ###
    ### the global sparsity we compute is the sparsity of at the Conv and Linear layers, not include BN layers ###
    total = 0
    num_zero = 0
    for module in net.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            [zero_now, total_now] = compute_module_sparsity(module)
            num_zero += zero_now
            total += total_now
    sparsity = num_zero / total
    return num_zero, total, sparsity


def update_model_by_mask(train_model):
    ### After pruning, we will update the parameters by mask in the network by "removing" that weight which is masked###
    ### we just remove weight, if needed, we can remove bias as well  ###
    for module in train_model.model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            try:
                prune.remove(module, "weight")
            except:
                pass


def compute_test_accuracy(train_model):
    ### compute accuracy ###
    with torch.no_grad():
        correct = 0
        total = 0
        for data in train_model.testloader:
            train_model.model.eval()
            images, labels = data
            images, labels = images.to(train_model.device), labels.to(train_model.device)
            outputs = train_model.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Current test_acc: %.3f%%' % (100 * correct / total))
    return (100 * correct / total)


def iterative_pruning(train_model, amount_perstep, final_sparsity):
    ### do interative purning ###
    sparsity = 0
    diff = []
    accurancy = []
    Sparity = []
    while sparsity < final_sparsity:
        # compute the accuracy before and just after the pruning
        test_accuracy_before = compute_test_accuracy(train_model)
        one_shot_global_pruning(train_model, amount_perstep)
        _, _, sparsity = compute_global_sparsity(train_model.model)
        print("Current Sparsity is: ", sparsity)
        test_accuracy_after = compute_test_accuracy(train_model)
        print("training")
        train_model.train()
        test_accuracy = compute_test_accuracy(train_model)
        diff.append(test_accuracy_before - test_accuracy_after)
        accurancy.append(test_accuracy)
        Sparity.append(sparsity)
    return accurancy, Sparity, diff




if __name__ == "__main__":
    amount_ratio_list = [0.5, 0.75, 0.9]
    train_model = model_train(EPOCH=5, LR=0.001)
    train_model.loaddataset()
    print("Please choose iterative pruning or one-shot pruning. 0 means iterative and 1 means one-shot")
    one_shot = int(sys.stdin.readline().strip())
    while one_shot != 0 and one_shot != 1:
        print('Error, please input again!')
        one_shot = int(sys.stdin.readline().strip())

    accurancy = {}
    sparity = {}
    accurancy_drop = {}



    if one_shot == 1:
        # one shot pruning
        print("One-shot pruning...")
        train_model.args.net = './model/oneshot_best_model.pth'

        for amount_ratio in amount_ratio_list:
            train_model.reload_model()
            test_accuracy_before = compute_test_accuracy(train_model)
            one_shot_global_pruning(train_model, amount_ratio)
            _, _, sparsity = compute_global_sparsity(train_model.model)
            print("Current Sparsity is: ", sparsity)
            test_accuracy_after = compute_test_accuracy(train_model)
            print("training")
            train_model.train()
            test_accuracy = compute_test_accuracy(train_model)
            accurancy_drop[amount_ratio] = test_accuracy_before - test_accuracy_after
            accurancy[amount_ratio] = test_accuracy
            sparity[amount_ratio] = sparsity
        print(accurancy_drop, '\n', accurancy, '\n', sparity)
    else:
        # iterative pruning
        # since it is iterative so we can get results of sparsity = 0.5 & 0.75 in the progress of sparsity = 0.9
        print('Iterative pruning...')
        train_model.args.net = './model/iterative_best_model.pth'
        amount_ratio = amount_ratio_list[-1]
        train_model.reload_model()
        print("Please input the sparsity per iteration, it should be in range(0, 1)")
        s = float(sys.stdin.readline().strip())
        while s <= 0 or s >= 1:
            print("Error, please input again!")
            s = float(sys.stdin.readline().strip())
        accurancy[amount_ratio], sparity[amount_ratio], accurancy_drop[amount_ratio] = iterative_pruning(train_model, s, amount_ratio)
        print(accurancy_drop, '\n', accurancy, '\n', sparity)
