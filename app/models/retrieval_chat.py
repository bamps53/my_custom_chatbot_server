import pickle
import threading

from langchain import LLMChain, OpenAI, ConversationChain
from langchain.callbacks import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
# from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
#                                                      QA_PROMPT)
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain

from app.callbacks.streaming import ThreadedGenerator, ChainStreamHandler

with open("iv_vectorstore.pkl", "rb") as f:
    global vectorstore
    vectorstore = pickle.load(f)

from langchain.prompts.prompt import PromptTemplate

_template = """あなたはキーエンスの画像センサIV3専門のチャットボットです。以下にIV3の特徴、これまでの会話履歴、顧客の質問が与えられるので、今回答すべき質問を再度1文で言い換えてください。

IV3の概要:
キーエンスの画像判別センサであるIV3は、従来の画像判別センサに比べて、以下のような特徴があります。
- 撮像条件決め(絵作り)も検出条件決めも、有無・判別に特化したAIが自動で行います。
- 注目したい部分を決めて、OK/NG画像を最低1枚ずつ登録するだけで、専門的な知識や設定の手間・時間を必要とせず、簡単に導入・運用が可能となります。
- レンズ・照明も内蔵したオールインワン設計のため、わずらわしい機器選定も不要で、外乱光の影響を受けずに検出課題の即解決を実現します。
- 高性能CPUを搭載した小型アンプで設定操作が完了するため、高性能PCも不要です。
- 超小型ヘッドが設置距離50~2000mmの超ワイドレンジに対応し、視野も最大1822x1364mmの超広角での検出が可能なため、あらゆるシーンで活躍します。
- IV3は、従来の画像判別センサに比べて、より簡単に、より確実に、より幅広いシーンで検出を行うことができる、画期的な製品です。

会話履歴:
{chat_history}

顧客の質問: {question}

あなたが言い換えた質問:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """あなたはキーエンスの画像センサIV3専門のチャットボットです。以下にIV3の特徴、回答に有用な情報、顧客からの質問が与えられるので、質問に丁寧に回答してください。もし答えがわからない場合は素直にそう述べてください。存在しない回答を作り上げることは絶対にやめてください。手順を説明する場合はステップバイステップで箇条書きにしてください。

IV3の概要:
キーエンスの画像判別センサであるIV3は、従来の画像判別センサに比べて、以下のような特徴があります。
- 撮像条件決め(絵作り)も検出条件決めも、有無・判別に特化したAIが自動で行います。
- 注目したい部分を決めて、OK/NG画像を最低1枚ずつ登録するだけで、専門的な知識や設定の手間・時間を必要とせず、簡単に導入・運用が可能となります。
- レンズ・照明も内蔵したオールインワン設計のため、わずらわしい機器選定も不要で、外乱光の影響を受けずに検出課題の即解決を実現します。
- 高性能CPUを搭載した小型アンプで設定操作が完了するため、高性能PCも不要です。
- 超小型ヘッドが設置距離50~2000mmの超ワイドレンジに対応し、視野も最大1822x1364mmの超広角での検出が可能なため、あらゆるシーンで活躍します。
- IV3は、従来の画像判別センサに比べて、より簡単に、より確実に、より幅広いシーンで検出を行うことができる、画期的な製品です。

回答に有用な情報:
{context}

顧客の質問:
{question}
あなたの回答:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


class RetrievalConversationChat:
    def __init__(self, history):
        self.memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
        self.set_memory(history)

    def set_memory(self, history):
        for message in history:
            if message.role == 'assistant':
                self.memory.chat_memory.add_ai_message(message.content)
            else:
                self.memory.chat_memory.add_user_message(message.content)

    def generator(self, user_message):
        g = ThreadedGenerator()
        threading.Thread(target=self.llm_thread, args=(g, user_message)).start()
        return g

    def llm_thread(self, g, user_message):
        try:
            question_gen_llm = OpenAI(
                temperature=0,
                verbose=True,
                max_tokens=1000,
            )
            streaming_llm = OpenAI(
                streaming=True,
                verbose=True,
                temperature=0,
                max_tokens=1000,
                callback_manager=CallbackManager([ChainStreamHandler(g)]),
            )

            question_generator = LLMChain(
                llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True
            )
            doc_chain = load_qa_chain(
                streaming_llm, chain_type="stuff", prompt=QA_PROMPT, verbose=True
            )

            qa = ConversationalRetrievalChain(
                retriever=vectorstore.as_retriever(),
                combine_docs_chain=doc_chain,
                question_generator=question_generator,
                # memory=self.memory,
                max_tokens_limit=4000-2000,
                verbose=True,
            )
            qa({"question": user_message, 'chat_history': []})
        finally:
            g.close()
