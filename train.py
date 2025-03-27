import argparse
import os
import copy
import numpy as np
from torch import Tensor
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, save_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--train-file', type=str, required=True,)  
    parser.add_argument('--eval-file', type=str, required=True)  
    parser.add_argument('--outputs-dir', type=str, required=True)   
    parser.add_argument('--scale', type=int, default=3) 
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--batch-size', type=int, default=16) 
    parser.add_argument('--num-workers', type=int, default=0)  
    parser.add_argument('--num-epochs', type=int, default=400)  
    parser.add_argument('--seed', type=int, default=123) 
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    # Enables faster training
    cudnn.benchmark = True
    # Enables faster training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #  Fix random seed for reproducibility
    torch.manual_seed(args.seed)
    model = SRCNN().to(device)

    # 恢复训练，从之前结束的那个地方开始
    # model.load_state_dict(torch.load('outputs/x3/epoch_173.pth'))
    # # Mean Squared Error (MSE) loss
    criterion = nn.MSELoss()

    #Lower LR for final layer
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    
    #Pinned memory locks the data in RAM, allowing GPU to access it directly, making the transfer much faster.
    #Drops the last batch if it has fewer images than the batch size. [Drop_last]
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,shuffle=True,                                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    lossLog = []
    psnrLog = []

    
    # for epoch in range(args.num_epochs):
    #don't need batching because we aren't updating weights.
    for epoch in range(1, args.num_epochs + 1):  
        model.train()
        epoch_losses = AverageMeter()
        #total=(len(train_dataset) - len(train_dataset) % args.batch_size) ensures that only complete batches are considered (avoiding leftover data).
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            # t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs - 1))
            t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs))

            for data in train_dataloader:
                # Get LR (inputs) and HR (labels) images
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)
                # Pass LR images through the model (SRCNN)
                preds = model(inputs) 
                # Compute loss
                loss = criterion(preds, labels)

                # Track loss
                epoch_losses.update(loss.item(), len(inputs)) #converted to float
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Displays live loss updates on the tqdm progress bar.
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                
        # Saves the loss value for this epoch into a list (lossLog)
        lossLog.append(np.array(epoch_losses.avg))
        # Saves all recorded loss values into "lossLog.txt" for future reference.
        np.savetxt("C:/Users/User/Desktop/div2k_train_val/outputs/lossLog.txt", lossLog)
        # resuming training from a checkpoint.
        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Disable gradient computation during evaluation
            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0) #All pixel values will now be between [0,1] , to avoid possibility of values failling out of range [Prevents Issues When Converting Back to RGB]

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        psnrLog.append(Tensor.cpu(epoch_psnr.avg))
        np.savetxt('C:/Users/User/Desktop/div2k_train_val/outputs/psnrLog.txt', psnrLog)
        
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        # Save Loss and PSNR plots
        save_plot(lossLog, lossLog, psnrLog, psnrLog)

        #	Updates & saves whenever a new best PSNR is found.
        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
        torch.save(best_weights, os.path.join(args.outputs_dir, 'C:/Users/User/Desktop/div2k_train_val/outputs/best.pth'))

    #Ensures the final best model is saved after training.
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'C:/Users/User/Desktop/div2k_train_val/outputs/best.pth'))
    
    
    
    
    
    
    
    
    
    