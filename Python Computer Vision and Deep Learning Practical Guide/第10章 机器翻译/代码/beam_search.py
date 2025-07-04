import torch
from config import MAX_LENGTH, device, regrex, EOS_token, SOS_token
from data import train_data
from queue import PriorityQueue
import operator


class BeamSearchNode:
    def __init__(self, hidden_state, previous_node, word_id, log_prob, length):
        self.hidden_state = hidden_state
        self.previous_node = previous_node
        self.word_id = word_id
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        # 计算该节点的分数，使用length做惩罚，越长概率越低
        reward = 0
        return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward


def beam_search(encoder, decoder, sentence, max_length=MAX_LENGTH):
    topk = 2  # 最终得到的句子数量
    beam_width = 10
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
        # eos节点
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # 创建第一个node
        node = BeamSearchNode(
            length=1,
            log_prob=0,
            word_id=de_inp,
            hidden_state=de_hid,
            previous_node=None,
        )
        nodes = PriorityQueue()
        nodes.put((-node.eval(), node))

        qsize = 1
        # 开始beam search
        while True:
            if qsize > 2000:
                print("qsize is larger than 2000")
                break
            score, n = nodes.get()
            de_inp = n.word_id
            de_hid = n.hidden_state

            # 如果得到了EOS_token，就纪录到endnodes中
            # 如果得到的endnodes足够多，则停止预测
            if n.word_id.item() == EOS_token and n.previous_node != None:
                endnodes.append((score, n))
                # print("endnodes num ", len(endnodes))
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            de_out, de_hid, _ = decoder(de_inp, de_hid, en_out)

            # 找到下一个节点的前beam_width个概率最大的词汇
            log_prob, indices = torch.topk(de_out, beam_width)
            # 统计下一节点情况
            next_nodes = []

            # 分别计算这个beam_width个节点的log概率（到根节点的整个句子概率）
            for new_k in range(beam_width):
                de_t = indices[0][new_k].view(-1, 1)
                log_p = log_prob[0][new_k].item()
                # 计算概率时加上前一个节点的log概率，即可得到到根节点（首个单词）的整个句子概率
                node = BeamSearchNode(
                    de_hid, n, de_t, n.log_prob + log_p, n.length + 1
                )
                score = -node.eval()
                next_nodes.append((score, node))

            # 加入队列
            for i in range(len(next_nodes)):
                score, nn = next_nodes[i]
                nodes.put((score, nn))
            qsize += len(next_nodes) - 1

        # 如果一直没有预测到eos，就从队列末尾取topk个节点回溯
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        # 从endnodes中的节点开始回溯
        utterances = []
        results = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.word_id)
            while n.previous_node != None:
                n = n.previous_node
                utterance.append(n.word_id)
            utterance = utterance[::-1]
            utterances.append(utterance)
            result = [
                train_data.zh_data.index2word[r.item()] for r in utterance
            ]
            results.append(result)
    return results, utterances
