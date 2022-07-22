import os ,torch
import argparse
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
import logging
from Dataload import ABAWDataSet
from norm import pNorm
from GUS_model import *
from until import *
from loss import *

logger = logging.getLogger('test_logger')
logger.setLevel(logging.DEBUG)
path_time = str(get_now_time())
isExists = os.path.exists(path_time)
if not isExists:
    print(path_time)
    os.makedirs(path_time)
    
test_log = logging.FileHandler(path_time+'/base.log','a',encoding='utf-8')
test_log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s')
test_log.setFormatter(formatter)
logger.addHandler(test_log)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/xj/Data/FERdata/2/', help='Raf-DB dataset path.')
    parser.add_argument('--data_lable', type=str, default='/home/xj/Data/FERdata/2/', help='Raf-DB dataset path.')
    parser.add_argument('--pretrained', type=str, default="", help='Pretrained weights')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="sgd", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Drop out rate.')
    return parser.parse_args()


def mean_f1(preds,targets):
    f1=[]
    temp_exp_pred = np.array(preds)
    temp_exp_target = np.array(targets)
    temp_exp_pred = torch.eye(6)[temp_exp_pred]
    temp_exp_target = torch.eye(6)[temp_exp_target]
    for i in range(0, 6):
        exp_pred = temp_exp_pred[:, i]
        exp_target = temp_exp_target[:, i]
        f1.append(f1_score(exp_pred, exp_target))
    print(f1)
    logger.info(str(f1))
    return np.mean(f1)

def calculate_loss(criterion, out, y, norm=None, lamb=None, tau=None, p=None,is_sparse =True):
    if is_sparse:
        out = F.normalize(out, dim=1)
        loss = criterion(out / tau, y) + lamb * norm(out / tau, p)
    else:
        loss = criterion(out, y)
    return loss
    
def run_training(threshold):
    
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    datapathlist = [args.data_path ]
    labelpathlist = [args.data_lable]

    modelfer = Res18Feature(pretrained = True, drop_rate = args.drop_rate)

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained) 
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['model_state_dict']
        modelfer.load_state_dict(pretrained_state_dict, strict = False)
  
    train_dataset = ABAWDataSet(datapathlist, labelpathlist, phase = 'train')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)
                                 
    val_dataset = ABAWDataSet(datapathlist, labelpathlist, phase = 'val')

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    params = modelfer.parameters()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,weight_decay = 1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay = 1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

    modelfer = modelfer.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    fl = FocalLoss()
    norm =pNorm(0.1)
    max_acc = 0
    for i in range(1, args.epochs):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        modelfer.train()
        for batch_i, (imgs, targets, path, indexes) in tqdm(enumerate(train_loader)):
            imgs, targets = imgs.to("cuda"), targets.to("cuda")
            iter_cnt += 1
            outputs, feature = modelfer(imgs, "train", threshold) 

            #loss_c = criterion(outputs, targets)
            #loss_mae=mae(outputs, targets)
            #loss_fl=fl(outputs, targets)
            #loss_nfl=nfl(outputs, targets)
            #loss_rce=rce(outputs, targets)
            #loss = 0.5*loss_nfl+0.5*loss_rce
            loss = calculate_loss(fl, outputs, targets, norm, 0.1, 0.5, 0.1)
            #loss = loss_c + loss_fl
            #loss = 0.1*loss_rce+20*loss_nfl
            #print(loss_c,loss_nfl,loss_rce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num


        if True:
            scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))
        logger.info('[Epoch %d] Training accuracy: %.4f. Loss: %.3f\n' % (i, acc, running_loss))

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            modelfer.eval()
            f1 =[]
            _p =[]
            _t =[]
            for batch_i, (imgs, targets, _, _) in enumerate(val_loader):
                outputs,feature = modelfer(imgs.cuda(), "val", threshold)
                targets = targets.cuda()
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                iter_cnt+=1
                _, predicts = torch.max(outputs, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)
                for p,t in zip(predicts,targets):
                    _p.append(p.cpu())
                    _t.append(t.cpu())
            all_f1= mean_f1(_p,_t)
            running_loss = running_loss/iter_cnt   
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f. f1:%.4f" % (i, acc, running_loss,all_f1))
            logger.info("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f. f1:%.4f \n" % (i, acc, running_loss,all_f1))
            if max_acc < acc:
                max_acc = acc
                torch.save({'iter': i,
                            'model_state_dict': modelfer.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join(path_time,"Res18_"+str(i)+"_ACC_"+str(acc)+ ".pth"))
                print('Model saved.')
     
            
if __name__ == "__main__":
    run_training(0.5)
