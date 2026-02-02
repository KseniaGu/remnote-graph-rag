import re
from pathlib import Path
from typing import Optional

import requests
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from tqdm import tqdm

from backend.configs.constants import REMNOTE_IMAGE_HOST_MARKER, HAS_PARSED_URL_MARKER, \
    HAS_UNPARSED_URL_MARKER, EXCLUDED_EMBED_METADATA_KEYS, EXCLUDED_LLM_METADATA_KEYS, DOCUMENT_PARSER_LOGGING
from backend.configs.enums import PDFParsingStatus, ImageParsingStatus, StorageType, ExternalSourceUrlType, ContentType
from backend.configs.paths import PathSettings
from backend.configs.storage import StorageSettings
from backend.data_processing.ocr import PaddleOCRPipeline
from backend.data_processing.utils import clean_text, save_pdf_by_url, search_for_url, save_image_by_url, \
    save_text_by_url
from backend.knowledge_graph.storage import KnowledgeGraphStorage
from backend.utils.helpers import get_logger

logger = get_logger(DOCUMENT_PARSER_LOGGING)


class RemNoteParser:
    """Parser for RemNote markdown files with hierarchical structure and external URL handling.
    
    Parses RemNote exported markdown files into hierarchical document nodes, handles external
    URLs (PDFs, images, text), and maintains parent-child relationships between nodes.
    """

    def __init__(self, path_settings: PathSettings, storage_settings: StorageSettings):
        """Initializes the RemNote parser.
        
        Args:
            path_settings: Path configuration settings.
            storage_settings: Storage configuration settings.
        """
        self.path_settings = path_settings

        self.splitter = SentenceSplitter()
        self.kg_storage = KnowledgeGraphStorage(
            path_settings, storage_settings, local_storage=path_settings.local_storage_dir
        )
        self.document_storage_type = storage_settings.document_storage.storage_type
        self.ocr_pipeline = PaddleOCRPipeline()

    def add_text_nodes(self, text_nodes: list[TextNode], allow_update: bool = True):
        """Adds text nodes to the document store.
        
        Args:
            text_nodes: List of text nodes to add.
            allow_update: Whether to allow updating existing nodes.
        """
        logger.info("Adding base document nodes")
        self.kg_storage.storage_context.docstore.add_documents(text_nodes, allow_update=allow_update)
        if self.document_storage_type == StorageType.local:
            self.kg_storage.storage_context.persist(persist_dir=str(self.path_settings.local_storage_dir))

    @staticmethod
    def check_headers(
            stripped_line: str,
            found_headers: set[int],
            indent_level: int,
            header_bonus: Optional[int]
    ) -> tuple[set[int], Optional[int], int]:
        """Checks for markdown headers and calculates depth level.
        
        Args:
            stripped_line: Stripped line of text.
            found_headers: Set of header levels found so far.
            indent_level: Current indentation level.
            header_bonus: Bonus depth from previous headers.
            
        Returns:
            Tuple of (updated found_headers, header_bonus, depth_level).
        """
        header_match = re.search(r"^[\s-]*(#+)", stripped_line)
        if header_match:
            header_length = header_bonus = len(header_match.group(1))
            for i in reversed(range(1, header_length)):
                if i not in found_headers:
                    header_bonus -= 1

            found_headers.add(header_length)
            header_bonus -= 1  # We set # as 0 indent, ## as 1 indent and so on

            depth_level = indent_level + header_bonus
        else:
            bonus = header_bonus + 1 if header_bonus is not None else 0
            depth_level = indent_level + bonus
        return found_headers, header_bonus, depth_level

    @staticmethod
    def create_text_node(
            filename: str,
            line_number: int,
            raw_content: str,
            depth_level: int,
            has_unparsed_url: bool,
            text: str,
            node_stack: list[Optional[TextNode]]
    ) -> tuple[TextNode, list[Optional[TextNode]]]:
        """Creates a text node with hierarchical relationships.
        
        Args:
            filename: Source filename.
            line_number: Line number in the source file.
            raw_content: Raw content before cleaning.
            depth_level: Hierarchical depth level.
            has_unparsed_url: Whether the node contains unparsed URLs.
            text: Full context text with hierarchy.
            node_stack: Stack of parent nodes.
            
        Returns:
            Tuple of (new text node, updated node stack).
        """
        parts = [p.strip() for p in text.split(">")]
        metadata = {
            "source": filename,
            "line_number": line_number,
            "original_text": raw_content,
            "depth_level": depth_level,
            "path": parts[:-1]
        }
        # Add flag to parse urls further in self.parse_external_links() method
        if has_unparsed_url:
            metadata[HAS_UNPARSED_URL_MARKER] = has_unparsed_url

        new_node = TextNode(
            text=parts[-1],
            metadata=metadata,
            excluded_embed_metadata_keys=EXCLUDED_EMBED_METADATA_KEYS,
            excluded_llm_metadata_keys=EXCLUDED_LLM_METADATA_KEYS
        )
        if depth_level > 0 and node_stack:
            parent_node = next((n for n in reversed(node_stack) if n is not None), None)
            if parent_node:
                # Link Child -> Parent
                new_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=parent_node.node_id,
                    metadata={"title": parent_node.text}
                )

                # Link Parent -> Child
                if "child_ids" not in parent_node.metadata:
                    parent_node.metadata["child_ids"] = []
                    parent_node.relationships[NodeRelationship.CHILD] = []

                parent_node.metadata["child_ids"].append(new_node.node_id)
                parent_node.relationships[NodeRelationship.CHILD].append(
                    RelatedNodeInfo(node_id=new_node.node_id, metadata={"title": parts[-1]})
                )

        return new_node, node_stack

    def parse_markdown_file(self, file_path: Path, check_url: bool = False) -> list[TextNode]:
        """Reads a Markdown file and converts the data into context-aware document nodes.
        
        Args:
            file_path: Path to the markdown file.
            check_url: Whether to check for external URLs.
            
        Returns:
            List of parsed text nodes with hierarchical relationships.
        """
        node_stack, context_stack, nodes = [], [], []
        header_bonus = None
        filename = file_path.stem
        found_headers = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f):
                    has_unparsed_url = False
                    stripped_line = line.strip()

                    if not stripped_line or stripped_line.startswith("---"):
                        continue

                    # Simple base check for url
                    if check_url and "http" in stripped_line:
                        has_unparsed_url = True

                    # Calculate indentation level
                    indent_spaces = len(line) - len(line.lstrip())
                    indent_level = indent_spaces // 2

                    # Check line for headers
                    found_headers, header_bonus, depth_level = self.check_headers(
                        stripped_line, found_headers, indent_level, header_bonus
                    )

                    # Clean content
                    raw_content = stripped_line.lstrip("-*").strip()
                    cleaned_content = clean_text(raw_content)
                    if not cleaned_content.strip() and REMNOTE_IMAGE_HOST_MARKER in raw_content:
                        # Replace empty string with a mask
                        cleaned_content = ExternalSourceUrlType.image.value

                    if not raw_content:
                        continue

                    # Manage depth levels
                    if depth_level < len(node_stack):
                        node_stack = node_stack[:depth_level]
                        context_stack = context_stack[:depth_level]

                    # In case indentation is not homogeneous (e.g. only # and ### are presented and ## is missed)
                    while len(node_stack) < depth_level:
                        placeholder = node_stack[-1] if node_stack else None
                        node_stack.append(placeholder)
                        context_stack.append("...")

                    context_stack.append(cleaned_content)

                    # Example: "Deep Learning > Neural Networks > Activation Functions > ReLU"
                    full_context_text = f"{filename} > " + " > ".join([c for c in context_stack if c != "..."])

                    new_node, node_stack = self.create_text_node(
                        filename, line_number, raw_content, depth_level, has_unparsed_url, full_context_text, node_stack
                    )

                    nodes.append(new_node)
                    node_stack.append(new_node)
        except Exception:
            logger.error(f'Exception while parsing "{file_path.stem}"', exc_info=True)
            pass

        return nodes

    @staticmethod
    def update_text_nodes(
            nodes: list[TextNode],
            parent_node_info: TextNode,
            documents_to_update: list[TextNode]
    ) -> tuple[list[TextNode], TextNode, list[TextNode]]:
        """Updates text nodes generated from parsed external links.
        
        Changes the source and connects nodes with their parent node.
        
        Args:
            nodes: List of nodes to update.
            parent_node_info: Parent node information.
            documents_to_update: List of documents to update.
            
        Returns:
            Tuple of (updated nodes, updated parent, documents to update).
        """
        childs = []
        for node in nodes:
            node.metadata['depth_level'] = node.metadata['depth_level'] + parent_node_info.metadata['depth_level'] + 1
            # Keep parent's line number for simplicity
            node.metadata['line_number'] = parent_node_info.metadata['line_number']
            node.metadata["parsed_from_external_url"] = True
            node.metadata["path"] = parent_node_info.metadata["path"] + node.metadata["path"]
            node.metadata['source'] = parent_node_info.metadata['source']

            if node.parent_node is None:
                node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=parent_node_info.node_id,
                    metadata={"title": parent_node_info.text}
                )
                childs.append(node)

        if "child_ids" not in parent_node_info.metadata:
            parent_node_info.metadata["child_ids"] = []
            parent_node_info.relationships[NodeRelationship.CHILD] = []

        parent_node_info.metadata["child_ids"].extend([node.node_id for node in childs])
        childs_related_info = [RelatedNodeInfo(node_id=node.node_id, metadata={"title": node.text}) for node in childs]
        parent_node_info.relationships[NodeRelationship.CHILD].extend(childs_related_info)
        parent_node_info.metadata.pop(HAS_UNPARSED_URL_MARKER)
        parent_node_info.metadata[HAS_PARSED_URL_MARKER] = True

        documents_to_update.extend(nodes)
        documents_to_update.append(parent_node_info)
        return nodes, parent_node_info, documents_to_update

    def parse_external_links(self):
        """Parses external URLs found in document nodes.
        
        Processes PDFs, images, and text files from external URLs and integrates
        them into the document hierarchy.
        """
        logger.info("Parsing external links")
        nodes_to_update = []
        nodes_iter = self.kg_storage.storage_context.docstore.docs.items()

        for node_id, node_info in tqdm(nodes_iter, desc="Updating nodes with external urls"):
            if not node_info.metadata.get(HAS_UNPARSED_URL_MARKER):
                continue

            try:
                mkd_file_path = None
                name, url = search_for_url(node_info.metadata['original_text'])

                response = requests.get(url)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")

                if ContentType.application_json.value in content_type:
                    status, saved_file_path = save_pdf_by_url(response, name, url, self.path_settings.parsed_pdfs_dir)

                    if status in (PDFParsingStatus.file_exists, PDFParsingStatus.success):
                        mkd_file_path = self.ocr_pipeline.parse_pdf(saved_file_path)

                elif ContentType.image.value in content_type:
                    status, saved_file_path = save_image_by_url(
                        response, content_type, name, url, self.path_settings.parsed_images_dir
                    )
                    if status in (ImageParsingStatus.file_exists, ImageParsingStatus.success):
                        mkd_file_path = self.ocr_pipeline.parse_image(saved_file_path)

                elif any(x.value in content_type for x in ContentType if x.name.startswith("text")):
                    _, mkd_file_path = save_text_by_url(name, url, self.path_settings.parsed_texts_dir)

                if mkd_file_path:
                    nodes = self.parse_markdown_file(mkd_file_path, check_url=False)
                    if nodes:
                        nodes, node_info, nodes_to_update = self.update_text_nodes(nodes, node_info, nodes_to_update)

            except Exception:
                logger.error(f'Exception while parsing external url ({url})', exc_info=True)

        self.add_text_nodes(nodes_to_update, allow_update=True)

    def postprocess(self):
        """Base postprocessing pipeline.

        Performs the following steps:
            1. Some documents contain empty text, either due to unreadable content or URLs. If there is url that we
               haven't parsed, we bring back removed url to the node text. If the node text is empty, but there are no
               URLs, we ensure that the node does not have any child nodes and remove it from the store.
            2. We add a final number of child nodes to the metadata, so that this information will be available to LLM.
        """
        logger.info("Postprocessing document nodes")
        nodes_to_update, deleted_nodes_num = [], 0
        nodes = self.kg_storage.storage_context.docstore.docs

        for node_id, node_info in tqdm(nodes.items(), desc="Postprocessing documents"):
            changed = False

            # Add number of child nodes
            if node_info.child_nodes:
                node_info.metadata["num_of_child_nodes"] = len(node_info.child_nodes)
                changed = True

            # Bring back cleaned urls if they were not parsed
            if not node_info.text.strip():
                original_text = node_info.metadata["original_text"]

                if node_info.metadata.get(HAS_UNPARSED_URL_MARKER):
                    node_info.text = original_text.strip()
                    changed = True
                else:
                    # changed=False here means document doesn't have any child nodes
                    if not changed:
                        if node_info.parent_node:
                            parent_node = nodes.get(node_info.parent_node.node_id)
                            child_nodes_updated = [n for n in parent_node.child_nodes if n.node_id != node_id]
                            parent_node.metadata["child_ids"].remove(node_id)
                            parent_node.relationships[NodeRelationship.CHILD] = child_nodes_updated
                            nodes_to_update.append(parent_node)
                        self.kg_storage.storage_context.docstore.delete_document(node_id)
                        deleted_nodes_num += 1

            if changed:
                nodes_to_update.append(node_info)

        self.add_text_nodes(nodes_to_update, allow_update=True)
        logger.info(f"Number of deleted nodes: {deleted_nodes_num}")
        logger.info(f"Number of updated nodes: {len(nodes_to_update)}")

    def get_text_nodes(self) -> list[TextNode]:
        """Retrieves and parses all markdown files from raw data directory.
        
        Returns:
            List of all parsed text nodes.
        """
        logger.info("Getting base document nodes from raw data .md files")
        raw_dir = Path(self.path_settings.raw_data_dir)

        all_nodes = []
        # Assuming all RemNote data is stored as .md files. If other data presented, add appropriate parser
        md_files = list(raw_dir.rglob("*.md"))

        for md_file in tqdm(md_files, desc="Parsing Markdown files"):
            nodes = self.parse_markdown_file(md_file, check_url=True)
            all_nodes.extend(nodes)

        return all_nodes

    def run(self):
        """Runs the whole parsing pipeline."""
        if self.kg_storage.storage_context.docstore.docs:
            logger.info("Documents are already stored in the docstore")
            return

        self.add_text_nodes(self.get_text_nodes())
        self.parse_external_links()
        self.postprocess()

        assert self.kg_storage.storage_context.docstore.docs, "No docs found in the docstore"
