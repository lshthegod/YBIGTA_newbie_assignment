# 구현하세요!

def load_corpus() -> list[str]:
    """
    공개된 'ag_news' 데이터셋을 불러와 코퍼스로 변환하는 함수.

    Returns:
        list[str]: 코퍼스 문장 리스트
    """
    from datasets import load_dataset
    import random

    dataset = load_dataset("ag_news")
    corpus: list[str] = []
    for split in dataset.keys():
        texts = dataset[split]["text"]
        sampled_texts = random.sample(texts, 100)
        corpus.extend(sampled_texts)
    return corpus