########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "binidx":
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

            if args.data_file.endswith('/'):
                d_all = []
                for p in os.listdir(args.data_file):
                    if p.endswith(".idx"):
                        d_all += [p[:-4]]
                d_all.sort()
                rank_zero_info(d_all)
                exit(0)
            else:
                self.data = MMapIndexedDataset(args.data_file)
                self.data_size = len(self.data._bin_buffer) // 2
                rank_zero_info(f"Data has {self.data_size} tokens.")

            if args.my_qa_mask > 0:
                self.data_pile = MMapIndexedDataset('/fsx/BlinkDL/pile/pile_20B_tokenizer_text_document')
                self.data_pile_size = len(self.data_pile._bin_buffer) // 2

            if args.my_pile_stage > 0:
                # assert self.data_size == 332115325534 and self.vocab_size == 50277
                self.samples_per_epoch = args.epoch_steps * args.real_bsz
                assert self.samples_per_epoch == 40320
                rank_zero_info(f"########## Pile 20b-tokenized stage {args.my_pile_stage} ##########")
                dataset_slot = self.data_size // args.ctx_len
                if args.my_pile_stage != 4:
                    assert MaybeIsPrime(args.magic_prime)
                    assert args.magic_prime % 3 == 2
                    assert args.magic_prime / dataset_slot > 0.99 and args.magic_prime / dataset_slot <= 1
        elif args.data_type == "numpy":
            self.data = np.load(args.data_file).astype("int")
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens.")
            if args.pile_shuffle:
                indices = np.arange(0, self.data_size - args.ctx_len, args.pile_shuffle_step)
                self.pile_shuffle_indices = indices
                self.pile_shuffle_i = 0
                rank_zero_info(f"Data has {len(indices)} pile shuffle indices.")
        elif args.data_type == "uint16":
            self.data = np.fromfile(args.data_file, dtype=np.uint16).astype("int32").reshape(-1, args.my_sample_len)
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            self.data_size = self.data.shape[0]
            rank_zero_info(f"Data has {self.data_size} samples.")
        elif args.data_type == "wds_img":
            self.vocab_size = -1
            self.data_size = -1
            self.data = None
            self.error_count = 0
        else:
            if args.data_type == "dummy":
                rank_zero_info("Building dummy data...")
                self.data = ""
                for i in range(100000):
                    aa = (i) % 10000
                    bb = (i * i) % 10000
                    cc = aa + bb
                    self.data += f".{aa}+{bb}={cc}."
            else:
                self.data = open(args.data_file, "r", encoding=args.data_type).read()
            rank_zero_info("Building token list...")
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            # rank_zero_info()
            # for u in unique:
            #     print(u, end=' ')
            # rank_zero_info('\n\n')
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-16le") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        if args.data_type == "wds_img":
            def init_wds(self, bias=0):
                def identity(x):
                    return x            
                import webdataset as wds
                import torchvision.transforms as transforms
                # img_transform = transforms.Compose(
                #     [transforms.CenterCrop(256)]
                # )
                img_transform = transforms.Compose([
                    transforms.CenterCrop(512),
                    transforms.Resize((args.my_img_size))
                ])
                self.data_raw = wds.WebDataset(args.data_file, resampled=True).shuffle(10000, initial=1000, rng=random.Random(epoch*100000+rank+bias*1e9)).decode("torchrgb").to_tuple("jpg", "json", "txt").map_tuple(img_transform, identity, identity)
                for pp in self.data_raw.pipeline:
                    if 'Resampled' in str(pp):
                        pp.deterministic = True
                        def worker_seed():
                            return rank*100000+epoch+bias*1e9
                        pp.worker_seed = worker_seed
                self.data = iter(self.data_raw)
                # print(f"WebDataset loaded for rank {rank} epoch {epoch}")
            if self.data == None:
                init_wds(self)
            trial = 0
            while trial < 10:
                try:
                    dd = next(self.data) # jpg, json, txt
                    break
                except:
                    print(f'[dataloader error - epoch {epoch} rank {rank} - trying a new shuffle]')
                    self.error_count += 1
                    init_wds(self, self.error_count)
                    trial += 1
                    pass
            # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} {dd[2]}")
            # with open(f"sample_{rank}.txt", "a", encoding="utf-8") as tmp:
            #     tmp.write(f"epoch {epoch} idx {idx} rank {rank}/{world_size} {int(dd[1]['key'])}\n")
            return dd[0], dd[2]
        else:
            if args.data_type == "uint16":
                i = np.random.randint(0, self.data_size-1)
                dix = self.data[i]
                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)
            else:
                ctx_len = args.ctx_len
                req_len = ctx_len + 1
                magic_prime = args.magic_prime
                data = self.data

                if args.my_pile_stage > 0 and args.my_pile_stage != 4:
                    ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

                    if args.my_qa_mask > 0:
                        ii_orig = ii
                        if ii % 2 == 0:
                            ii = (ii // 2) * args.magic_prime
                            if args.ctx_len == 1024:
                                magic_prime = 324331313
                            elif args.ctx_len == 2048:
                                magic_prime = 162165671
                            elif args.ctx_len == 4096:
                                magic_prime = 81082817
                            data = self.data_pile
                        else:
                            ii = ii // 2

                    factor = (math.sqrt(5) - 1) / 2
                    factor = int(magic_prime * factor)
                    i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
                    if (args.my_qa_mask == 0) or (data == self.data_pile):
                        i = i + args.my_pile_shift
                    # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")
                elif args.pile_shuffle:
                    if self.pile_shuffle_i == 0:
                        np.random.default_rng((1 + rank) * args.pile_shuffle_seed).shuffle(self.pile_shuffle_indices)
                    i = self.pile_shuffle_indices[self.pile_shuffle_i]
                    self.pile_shuffle_i = (self.pile_shuffle_i + 1) % len(self.pile_shuffle_indices)
                else:
                    # cheat: pick a random spot in dataset
                    i = np.random.randint(0, self.data_size - req_len)

                if args.data_type == "binidx":
                    dix = data.get(idx=0, offset=i, length=req_len).astype(int)
                elif args.data_type == "numpy":
                    dix = data[i : i + req_len]
                else:
                    dix = [self.stoi[s] for s in data[i : i + req_len]]

                if args.my_qa_mask == 1:
                    if data == self.data_pile:
                        z = [1] * ctx_len
                    else:
                        z = [0] * ctx_len
                        z_sum = 0
                        isGood = False
                        for i in range(3, ctx_len):
                            if dix[i] == 27 and dix[i-1] == 34 and dix[i-2] == 187 and dix[i-3] == 187:
                                isGood = True
                            if dix[i] == 0:
                                isGood = False
                            if isGood:
                                z[i] = 1
                                z_sum += 1
                        if z_sum == 0:
                            z = [1] * ctx_len
                            i = np.random.randint(0, self.data_pile_size - req_len)
                            dix = self.data_pile.get(idx=0, offset=i, length=req_len).astype(int)
                    z = torch.tensor(z, dtype=torch.bfloat16)

                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)

                # if ii_orig < 50:
                #     # if rank == 1:
                #     print('rank', rank, 'i', ii_orig, ii, i, 'x', x[:5], '...', x[-5:])
                # else:
                #     exit(0)

                if args.my_qa_mask == 1:
                    return x, y, z

            return x, y
