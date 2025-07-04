import torch
import os.path as osp

from model import Encoder, Decoder, AttenDecoder
from data import train_data
from config import (
    MAX_LENGTH,
    device,
    regrex,
    EOS_token,
    HIDDEN_SIZE,
    ATTENTION,
    SOS_token,
    CHECKPOINT,
)

encoder = Encoder(train_data.en_data.num_words, HIDDEN_SIZE).to(device)
if ATTENTION:
    decoder = AttenDecoder(train_data.zh_data.num_words, HIDDEN_SIZE).to(device)
else:
    decoder = Decoder(train_data.zh_data.num_words, HIDDEN_SIZE).to(device)

ckpt = osp.join(CHECKPOINT, "attention:{}".format(ATTENTION))
if osp.exists(ckpt):
    ckpt_model = torch.load(ckpt)
    encoder.load_state_dict(ckpt_model["encoder"])
    decoder.load_state_dict(ckpt_model["decoder"])
    print("Model loaded ...")


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        sentence = regrex.sub(" ", sentence)

        # Encoder
        x = (
            torch.Tensor(
                [
                    [train_data.en_data.word2index[word]]
                    for word in sentence.strip().split()
                ]
            )
            .long()
            .to(device)
        )
        xb = torch.zeros((MAX_LENGTH, 1)).long().to(device)  # 输入扩展
        xb[: x.shape[0]] = x
        en_out, en_hid = encoder(xb.view(-1, 1))

        # Decoder
        de_inp = torch.Tensor([[SOS_token]]).to(device).long()
        de_hid = en_hid
        attens = torch.zeros((MAX_LENGTH, MAX_LENGTH)).to(device)
        r_tensor = []
        for t in range(MAX_LENGTH):
            bs = 1
            de_out, de_hid, atten = decoder(de_inp, de_hid, en_out)
            _, topi = de_out.topk(1)
            label = torch.argmax(de_out, dim=1)
            if topi.item() == EOS_token:
                break
            de_inp = torch.LongTensor([[topi[i][0] for i in range(bs)]])
            de_inp = de_inp.to(device)
            attens[t] = atten
            r_tensor.append(de_inp)
        results = [train_data.zh_data.index2word[r.item()] for r in r_tensor]
    return results, attens


if __name__ == "__main__":
    from beam_search import beam_search

    Beam = True
    while 1:
        sentence = input(">>")
        if Beam:
            result, _ = beam_search(encoder, decoder, sentence)
            for i, r in enumerate(result):
                print("Top {} result : ".format(i), "".join(r[1:-1]))
        else:
            result, _ = evaluate(encoder, decoder, sentence)
            print("".join(result))
