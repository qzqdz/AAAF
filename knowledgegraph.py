# -*- encoding:utf-8 -*-


from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import torch
from transformers import  AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
import ast
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity



from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

from gensim.corpora.dictionary import Dictionary
from gensim import corpora, models
from nltk.corpus import stopwords
import jieba
from collections import defaultdict


# v4 clear vision
class KnowledgeSentence(object):
    def __init__(self, k, device='cuda', kg=None,model_path=None,only_knowledge_mode=True,analysis_match=False,detector=None, num_topics=10, language='en', num_label=None):
        self.only_knowledge_mode = only_knowledge_mode
        self.analysis_match = analysis_match
        self.k = k
        self.t = 0
        self.train_embeddings = None
        self.test_embeddings = None




        # 假设kg是DataFrame格式，并有target、aspect、label三个字段
        if kg is None:
            # 创建一个示例DataFrame作为知识库
            data = {'target': ["target1", "target2"],
                    'aspect': ["aspect1", "aspect2"],
                    'label': [1, 0]}  # 示例数据，你需要用实际的数据替换
            self.knowledge_base = pd.DataFrame(data)
        else:
            # 确保传入的kg是DataFrame并且包含需要的列
            # assert isinstance(kg, pd.DataFrame), "kg needs to be a pandas DataFrame"
            # assert all(column in kg.columns for column in ['target', 'aspect', 'label']), "kg must contain 'target', 'aspect', and 'label' columns"
            # self.knowledge_base = kg



            if 'target' in kg.columns:
                self.knowledge_base = kg
            elif 'topic' in kg.columns:
                self.knowledge_base = kg.rename(columns={'topic': 'target'})


            # 检查是否有'label'列，如果没有，生成一个'label'列，所有值为-1
            if 'label' not in self.knowledge_base.columns:
                self.knowledge_base['label'] = 1

        if num_label:
            self.num_label = num_label
        else:
            self.num_label = len(self.knowledge_base['label'].unique())

        # 设置设备
        if device == 'cpu':
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 设置模型
        if model_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.encoder = AutoModel.from_pretrained(model_path).to(self.device)


        if self.only_knowledge_mode==False:
            if self.analysis_match:
                train_prompts = kg['aspect'].tolist()

            else:
                train_prompts = kg['aspect'].tolist()
        else:
            train_prompts = kg['aspect'].tolist()

        # print("kg['aspect'].tolist()")
        # print(kg['aspect'].tolist())

        self.kg_prompts = train_prompts
        self.kg_embeddings = self.compute_embeddings(train_prompts)
        self.detector = detector

        # --------------- topic modeling ------------------
        # 登入停用词
        self.language = language
        if self.language == 'en':
            english_stopwords = set(stopwords.words('english'))
            self.stopwords = english_stopwords
        elif self.language == 'zh':
            with open('stopword.txt','r',encoding='utf-8') as f:
                # 按行读取停用词
                chinese_stopwords = f.read().split('\n')
            self.stopwords = chinese_stopwords


        # 生成主题模型
        self.num_topics = num_topics

        self.lda_model = None
        self.dictionary = None
        self.topic_label_distribution = None
        self.prepare_lda_model_and_topic_label_distribution()


    def word_tokenize(self, text):
        if self.language == 'en':
            return [word for word in text.lower().split() if word not in self.stopwords]
        elif self.language == 'zh':
            return [word for word in jieba.cut(text) if word not in self.stopwords]  # 使用jieba进行中文分词
        else:
            raise ValueError("Unsupported language")



    def prepare_lda_model_and_topic_label_distribution(self):
        texts = self.knowledge_base['aspect'].apply(self.word_tokenize).tolist()


        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]

        self.lda_model = models.LdaModel(corpus, num_topics=self.num_topics, id2word=self.dictionary, passes=15,
                                         random_state=42)

        # 初始化主题-立场分布统计
        topic_label_count = defaultdict(lambda: defaultdict(int))
        for i, row in self.knowledge_base.iterrows():
            bow = self.dictionary.doc2bow(texts[i])
            topics = self.lda_model.get_document_topics(bow)
            main_topic = max(topics, key=lambda x: x[1])[0]
            topic_label_count[main_topic][row['label']] += 1

        # 计算每个主题的立场分布百分比
        self.topic_label_distribution = {topic: {label: count / sum(labels.values()) for label, count in labels.items()}
                                         for topic, labels in topic_label_count.items()}




    # 计算单个文本的token和嵌入
    def embed(self,text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(
            self.device)  # 将输入数据移动到 GPU
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu()  # 将结果移回 CPU

    # 批量embed函数
    def compute_embeddings(self,text_samples):
        embeddings = []
        for sample in text_samples:
            embeddings.append(self.embed(sample).detach().numpy())  # 添加 detach() 方法
        return embeddings





    # 原版选knn的方法
    def select_knn_examples_ori(self,x_test, train_data, k):

        if self.analysis_match==True:
            assert self.only_knowledge_mode==False, "Shut down the knowledge mode!"


        test_embedding = self.embed(x_test).unsqueeze(0).numpy()  # 将结果转换为 NumPy 数组

        # 使用kNN查找最近邻
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(self.kg_embeddings)

        distances, indices = nbrs.kneighbors(test_embedding)
        # 返回选定的示例


        selected_examples = [self.kg_prompts[i] for i in indices.flatten()]


        if self.only_knowledge_mode==False:
            selected_examples = [train_data[tex] for tex in selected_examples]

        return selected_examples

    # 修改select_knn_examples以适应DataFrame
    def select_knn_examples(self, x_test, k):
        test_embedding = self.embed(x_test).unsqueeze(0).numpy()

        # 使用kNN查找最近邻
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(self.kg_embeddings)

        distances, indices = nbrs.kneighbors(test_embedding)
        selected_indices = indices.flatten()
        # selected_examples = [self.knowledge_base.iloc[i]['aspect'] for i in indices.flatten()]  # 从DataFrame获取aspect

        return selected_indices




    def get_distances(self, x_test):
        train_prompts = list(self.knowledge_base.values())
        self.train_embeddings = self.compute_embeddings(train_prompts)
        self.test_embeddings = self.compute_embeddings(x_test)

        # Use kNN to find the distances to all knowledge pieces
        nbrs = NearestNeighbors(metric="euclidean").fit(self.train_embeddings)
        distances, idx = nbrs.kneighbors(self.test_embeddings, len(train_prompts))
        print('dis and dix')
        print(distances[0])
        print(idx[0])
        # assert False,'stop'
        return distances


    # 这个是原版的get_knowledge
    def get_knowledge1(self, sentences,output_file='./kg/test_triples.txt', is_test=False, label=None):
        # 这个函数返回与输入句子相关的知识
        # 在这个简单的例子中，我们只是返回整个知识库
        # 在实际应用中，你可能需要根据输入句子的内容来选择返回哪些知识
        knowledge = []

        if label is not None and len(sentences) != len(label):
            raise ValueError("The number of sentences must match the number of labels")


        # If labels are provided, train the threshold classifier
        if label is not None:
            print('start to find the threshold:')
            # Calculate distances for all sentences

            distances = self.get_distances(sentences)
            flattened_distances = distances.flatten() if self.k > 1 else distances.reshape(-1)
            optimal_threshold = self.train_threshold_classifier(flattened_distances, label)

            # optimal_threshold = self.train_threshold_classifier(distances, label)
            print('optimal_threshold is:')
            print(optimal_threshold)


        #     创建文件output_file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('')


        # for i, _ in enumerate(sentences):
        print('Matching the knowledge of sentence.')
        for i in tqdm(range(len(sentences)), desc='Processing sentences'):

            if is_test == True:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(str(label[i]))
                    f.write(',')
            # 输入的是一句话，输出的是这句话的所有知识，是一句一句遍历的
            comment = self.select_knn_examples(sentences[i], self.knowledge_base, self.k)


            knowledge.append(comment)

            print('---------------------------')
            print(f'sentences[{i}] is here:')
            print(sentences[i])
            print('the knowledge of sentences:')
            print(knowledge[-1])
            print('---------------------------')


        return knowledge

    # 配合阈值方法用的knn选择器
    def select_knn_examples_for_one(self, x_test, train_data, k):
        if self.analysis_match == True:
            assert self.only_knowledge_mode == False, "Shut down the knowledge mode!"

        if self.only_knowledge_mode == False:
            if self.analysis_match:
                train_prompts = list(train_data.keys())
            else:
                train_prompts = list(train_data.values())
        else:
            train_prompts = list(train_data.values())

        train_embeddings = self.compute_embeddings(train_prompts)
        test_embedding = self.embed(x_test).unsqueeze(0).numpy()  # Convert to NumPy array

        # Using kNN to find the nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_embeddings)
        distances, indices = nbrs.kneighbors(test_embedding)

        return distances, indices




    def get_knowledge(self, sentences,output_file='./kg/test_triples.txt', is_test=False, label=None):
        # 这个函数返回与输入句子相关的知识
        # 在这个简单的例子中，我们只是返回整个知识库
        # 在实际应用中，你可能需要根据输入句子的内容来选择返回哪些知识
        knowledge = []
        knowledge_labels = []  # 新增列表以存储知识标签
        topic_labels = []  # 新建列表以存储主题标签概率


        # 校验是否存在'label'列
        has_label = 'label' in self.knowledge_base.columns

        if label is not None and len(sentences) != len(label):
            raise ValueError("The number of sentences must match the number of labels")


        if has_label:
            # Calculate distances for all sentences

            unique_labels = set(self.knowledge_base['label'])

        # for i, _ in enumerate(sentences):
        print('Matching the knowledge of sentence.')
        for i in tqdm(range(len(sentences)), desc='Processing sentences'):


            # lda
            if self.lda_model:
                # 预处理输入句子
                # processed_sentence = [word for word in sentences[i].lower().split() if word not in self.stopwords]
                processed_sentence = self.word_tokenize(sentences[i])
                bow = self.dictionary.doc2bow(processed_sentence)
                # 使用LDA模型获取句子的主题分布
                topic_distribution = self.lda_model.get_document_topics(bow, minimum_probability=0)

                # 初始化句子的标签概率为0
                sentence_label_probs = np.zeros(self.num_label)

                for topic_id, topic_prob in topic_distribution:
                    if topic_id in self.topic_label_distribution:
                        # print(self.topic_label_distribution.shape)
                        # Ensure topic_label_probs is a fully-populated array with a probability for each label
                        # topic_label_probs = np.array(
                        #     [self.topic_label_distribution[topic_id].get(label, 0) for label in range(len(unique_labels))])

                        topic_label_probs = np.zeros(len(unique_labels))
                        # 填充已知的标签概率
                        for label, prob in self.topic_label_distribution[topic_id].items():
                            topic_label_probs[label] = prob
                        # 累加调整后的主题标签概率向量到句子的标签概率向量
                        sentence_label_probs += topic_prob * topic_label_probs

                # Normalize if sentence_label_probs sum is not zero
                if sentence_label_probs.sum() != 0:
                    sentence_label_probs /= sentence_label_probs.sum()

                topic_labels.append(sentence_label_probs.tolist())

            if 1==2:
                pass
            else:
                selected_indices = self.select_knn_examples(sentences[i], self.k)
                selected_knowledge = [self.kg_prompts[i] for i in selected_indices]
                knowledge.append(selected_knowledge)
                # train_prompts
                # 如果有'label'列，则提取对应的标签，否则使用默认标签-1
                if has_label:
                    selected_labels = [int(self.knowledge_base.iloc[idx]['label']) for idx in selected_indices]
                else:
                    selected_labels = [1] * len(selected_indices)  # 创建等长的[1]列表

                knowledge_labels.append(selected_labels)

                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write('---------------------------\n')
                    f.write(f'sentences[{i}] is here:\n')
                    f.write(sentences[i])
                    f.write('\nthe knowledge of sentences:\n')
                    f.write(str(knowledge[-1]))
                    f.write('\nthe topic of sentences:\n')
                    f.write(str(topic_labels[-1]))
                    f.write('\n---------------------------\n')

                # comment = self.select_knn_examples(sentences[i], self.k)
                # knowledge.append(comment)

            # case 分析
            # print('---------------------------')
            # print(f'sentences[{i}] is here:')
            # print(sentences[i])
            # print('the knowledge of sentences:')
            # print(knowledge[-1])
            # print('the topic of sentences:')
            # print(str(topic_labels[-1]))
            # print('---------------------------')




        return knowledge, knowledge_labels, str(self.knowledge_base['target'].iloc[0]), topic_labels









