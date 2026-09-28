from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
from dataclasses import dataclass
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


@dataclass
class _RssAuthor:
    name: str


@dataclass
class _RssResult:
    title: str
    authors: list[_RssAuthor]
    summary: str
    pdf_url: str
    entry_id: str

    def source_url(self) -> str:
        return self.pdf_url.replace("/pdf/", "/src/")


def _result_from_rss(entry: feedparser.FeedParserDict) -> _RssResult:
    """Use the Atom feed's metadata when the arXiv query API is unavailable."""
    paper_id = entry.id.removeprefix("oai:arXiv.org:")
    summary = entry.get("summary", "")
    if "Abstract:" in summary:
        summary = summary.split("Abstract:", 1)[1].strip()
    author_text = entry.get("author", entry.get("dc_creator", ""))
    authors = [_RssAuthor(name.strip()) for name in author_text.split(",") if name.strip()]
    return _RssResult(
        title=entry.title,
        authors=authors,
        summary=summary,
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        entry_id=f"https://arxiv.org/abs/{paper_id}",
    )


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult | _RssResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        rss_entries = [
            entry for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            rss_entries = rss_entries[:10]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(rss_entries))
        max_batch_retries = 5
        batch_retry_delay = 30
        for i in range(0, len(rss_entries), 20):
            batch_entries = rss_entries[i:i + 20]
            search = arxiv.Search(id_list=[
                entry.id.removeprefix("oai:arXiv.org:") for entry in batch_entries
            ])
            for attempt in range(max_batch_retries):
                try:
                    batch = list(client.results(search))
                    bar.update(len(batch))
                    raw_papers.extend(batch)
                    break
                except arxiv.HTTPError as exc:
                    if exc.status == 429 and attempt < max_batch_retries - 1:
                        wait = batch_retry_delay * (attempt + 1)
                        logger.warning(f"arXiv API 429 on batch {i // 20}, retry {attempt + 1}/{max_batch_retries} in {wait}s")
                        sleep(wait)
                    elif exc.status in (406, 429):
                        logger.warning(
                            f"arXiv API HTTP {exc.status} on batch {i // 20}; "
                            "using Atom feed metadata for this batch"
                        )
                        batch = [_result_from_rss(entry) for entry in batch_entries]
                        bar.update(len(batch))
                        raw_papers.extend(batch)
                        break
                    else:
                        raise
            if i + 20 < len(rss_entries):
                sleep(3)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult | _RssResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult | _RssResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult | _RssResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult | _RssResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
