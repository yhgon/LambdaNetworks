import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time 
import argparse

import os
from preprocess import load_data
from model import ResNet18, ResNet50, LambdaResNet18,  LambdaResNet50, LambdaResNet152,  LambdaResNet200, LambdaResNet270, LambdaResNet350,  LambdaResNet420, get_n_params


def parse_args(parser):
    parser.add_argument('--batch_size',      type=int,   default=128)
    parser.add_argument('--num_workers',     type=int,   default=4)
    parser.add_argument('--lr',              type=float, default=0.1)
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--momentum',        type=float, default=0.9)
    parser.add_argument('--cuda',            type=bool,  default=True)
    parser.add_argument('--epochs',          type=int,   default=310)
    parser.add_argument('--print_intervals', type=int,   default=100)
    parser.add_argument('--evaluation',      type=bool,  default=False)
    parser.add_argument('--checkpoints',     type=str,   default=None, help='model checkpoints path')
    parser.add_argument('--device_num',      type=int,   default=1)
    parser.add_argument('--model_name',      type=str,   default='LambdaResNet18')

    return parser   



def save_checkpoint(best_acc, model, optimizer, args, epoch):
    print('Best Model Saving...')
    #if args.device_num > 1:
    #    model_state_dict = model.module.state_dict()
    #else:
    #    model_state_dict = model.state_dict()
    model_state_dict = model.state_dict()    

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, os.path.join('checkpoints', 'checkpoint_model_best.pth'))


def _train(epoch, train_loader, model, optimizer, criterion, args):    
    model.train()
    tic_epoch = time.time()
    losses = 0.
    acc = 0.
    total = 0.
    for idx, (data, target) in enumerate(train_loader):
        tic_iter = time.time()
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        _, pred = F.softmax(output, dim=-1).max(1)
        acc += pred.eq(target).sum().item()
        total += target.size(0)

        optimizer.zero_grad()
        loss = criterion(output, target)
        losses += loss
        loss.backward()
        optimizer.step()
        toc_iter = time.time()
        dur_iter = toc_iter - tic_iter
        iter_size = len(train_loader)
        
        

        if idx % args.print_intervals == 0 and idx != 0:
            print('\n[Epoch: {0:4d}] {} / {} | Loss: {1:.3f}, Acc: {2:.3f}, Correct {3} / Total {4} | {:4.2f}sec/iter'.format(
                epoch, idx, iter_size, losses / (idx + 1), acc / total * 100., acc, total, dur_iter), end='' )
    toc_epoch = time.time()  
    dur_epoch = toc_epoch - tic_epoch
    print(' {:4.2}sec/epoch'.format(dur_epoch), end='')
    


def _eval(epoch, test_loader, model, args):
    model.eval()

    acc = 0.
    with torch.no_grad():
        tic_eval = time.time()        
        for data, target in test_loader:            
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            _, pred = F.softmax(output, dim=-1).max(1)

            acc += pred.eq(target).sum().item()
        toc_eval = time.time()
        dur_eval = toc_eval - tic_eval            
        print('\n[Epoch: {0:4d}], Acc: {1:.3f} {:4.2f}sec/eval'.format(epoch, acc / len(test_loader.dataset) * 100., dur_eval), end='')

    return acc / len(test_loader.dataset) * 100.


def main(args):
    train_loader, test_loader = load_data(args)

    if args.model_name == 'ResNet18' :
        model = ResNet18()
    elif args.model_name == 'ResNet50' : 
        model = ResNet50()
    elif args.model_name == 'LambdaResNet18'   : 
        model = LambdaResNet18()
    elif args.model_name == 'LambdaResNet50'    :
        model = LambdaResNet50()
    elif args.model_name == 'LambdaResNet152'    :
        model = LambdaResNet152()
    elif args.model_name == 'LambdaResNet200'  :
        model = LambdaResNet200()
    elif args.model_name == 'LambdaResNet270'  :  
        model = LambdaResNet270()
    elif args.model_name == 'LambdaResNet350'   : 
        model = LambdaResNet350()
    elif args.model_name == 'LambdaResNet420' :
        model = LambdaResNet420()
    else :
        model = None
      

    print(model)
    print('{} Model Parameters: {:4.2f} M Params'.format(args.model_name, get_n_params(model)/1000000 ) )


    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    
    print("load checkpoint if needed")          
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if args.checkpoints is not None:
        checkpoints = torch.load(os.path.join('checkpoints', args.checkpoints))
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        start_epoch = checkpoints['global_epoch']
    else:
        start_epoch = 1

    if args.cuda:
        model = model.cuda()

    print("start iteration")        
    if not args.evaluation:
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)

        global_acc = 0.
        for epoch in range(start_epoch, args.epochs + 1):
            _train(epoch, train_loader, model, optimizer, criterion, args)
            best_acc = _eval(epoch, test_loader, model, args)
            if global_acc < best_acc:
                global_acc = best_acc
                save_checkpoint(best_acc, model, optimizer, args, epoch)
            print('Learning Rate: {}'.format(lr_scheduler.get_last_lr()), end='')
            lr_scheduler.step()
            print(' --> {}'.format(lr_scheduler.get_last_lr()))
    else:
        _eval(start_epoch, test_loader, model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LamdaResNet Training', 
                                     allow_abbrev=False) 
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    main(args)

