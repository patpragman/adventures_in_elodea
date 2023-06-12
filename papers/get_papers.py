import arxiv
import arxiv2bib
from chatgpttools import query_davinci
import json
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

search = arxiv.Search(
    query='convolutional neural network',
    max_results=100,
    sort_by=arxiv.SortCriterion.Relevance,
    sort_order=arxiv.SortOrder.Ascending
)



for result in search.results():
    query = f"""Hi there, I need your help, I'm summarizing critical papers in computer vision for
    a project.  I will pass you a summary of a paper, then I need you to reply whether you think it
    is worth including in my survey of the current literature.

    My research is in using computer vision to identify an invasive species.  Specifically, I am looking
    for foundational work on the topic of Computer Vision, algorithmic and machine learning papers, and
    papers that detail summaries of the field.  I'm particularly interested in papers about image classification.
    
    Please read the authors, the title, the abstract, and then please make your best decision based on how
    relevant to my work this paper will be.  I will download the papers that you suggest are relevant.  Please
    use your understanding of the field to assist in your decision making.  In your decision making process, weight
    well known authors much higher than authors that have little relevance to the field.
    
    Papers that are fundamental are more important here than ones that are merely interesting.  Please make
    a decision about the relevance, then reply with a JSON object that I can parse later.  The format should be
    exactly like this:

    {{"keep": true}} or {{"keep": false}} depending on whether you recommend I keep the paper or not.

    Authors:
    {result.authors}

    Title:
    {result.title}

    Summary:
    {result.summary}
    
    Thank you, this is very helpful for my research.
    """
    results = query_davinci(query)
    try:
        o = json.loads(results)
    except json.JSONDecodeError:
        print(o)

    if o["keep"]:
        print(result.title)
        result.download_pdf(dirpath="papers",
                            filename=f"{result.title.lower().replace(' ', '_')}.pdf")

        os.system(f'arxiv2bib {result.entry_id.split("/")[-1]} >> bibliography.tex')
