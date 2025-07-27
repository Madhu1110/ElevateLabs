from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_resumes(jd, resumes):
    corpus = [jd] + resumes
    vect = TfidfVectorizer()
    mat = vect.fit_transform(corpus)
    scores = cosine_similarity(mat[0:1], mat[1:])[0]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked
