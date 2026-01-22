# src/integrate_llm.py
from __future__ import annotations

from collections.abc import Iterator
from typing import TypedDict
import os
import subprocess
import time

import ollama
from ollama._types import ResponseError

from src.rag_retriever import RAGRetriever
from src.types import ChatMessage, RetrievedDocument


class OllamaChat:
    """
    RAG-enabled chat interface for Ollama models running in a Docker container.

    This class:
    - Ensures the Ollama container is running
    - Connects to the container endpoint
    - Maintains conversation history
    - Augments user queries with retrieved context
    - Supports both streaming and non-streaming responses
    
    All configuration (container name, host, port, volume) is passed via constructor.
    """

    def __init__(
        self,
        model: str,
        retriever: RAGRetriever,
        container_name: str,
        host: str = "http://localhost",
        port: int = 11434,
        data_volume: str = "ollama_data"
    ) -> None:
        """
        Initialize the Ollama chat client.

        Args:
            model: Ollama model name (e.g., "llama3.2:3b")
            retriever: RAG retriever instance used for context lookup
            container_name: Docker container name
            host: Ollama API endpoint host (e.g., http://localhost)
            port: Port for Ollama API
            data_volume: Docker volume name for persistent model storage
        """
        self.model = model
        self.retriever = retriever
        self.container_name = container_name
        self.port = port
        self.host = host
        self.data_volume = data_volume

        # Set Ollama SDK host to container endpoint
        os.environ["OLLAMA_HOST"] = f"{self.host}:{self.port}"

        # Start container if needed
        self._ensure_container_running()

        # Ensure the model exists inside the container
        self._ensure_model(self.model)

        # Conversation history
        self.history: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use the provided context. "
                    "If the answer is not in the context, say 'I don't know'."
                ),
            }
        ]

        self.welcome_message = (
            "Hello! I'm your RAG assistant. I can answer questions based on the "
            "data in the vector store. Type 'exit' or 'end' to quit."
        )
        self.history.append({"role": "assistant", "content": self.welcome_message})
        
    def _ensure_container_running(self) -> None:
        """Start Ollama Docker container if not already running, or start stopped container."""
        try:
            # Check if a container with the given name exists
            result = subprocess.run(
                ["docker", "ps", "-a", "-q", "-f", f"name={self.container_name}"],
                capture_output=True,
                text=True,
            )
            container_id = result.stdout.strip()

            if container_id:
                # Check if it is running
                result_running = subprocess.run(
                    ["docker", "ps", "-q", "-f", f"name={self.container_name}"],
                    capture_output=True,
                    text=True,
                )
                if result_running.stdout.strip():
                    print(f"Ollama container '{self.container_name}' already running.")
                    return
                else:
                    # Container exists but is stopped, start it
                    print(f"Starting existing Ollama container '{self.container_name}'...")
                    subprocess.run(["docker", "start", self.container_name], check=True)
                    print("Waiting 5 seconds for container to start...")
                    time.sleep(5)
                    return

            # Container does not exist, create a new one
            print(f"Creating and starting Ollama container '{self.container_name}'...")
            subprocess.run([
                "docker", "run", "-d",
                "--name", self.container_name,
                "-p", f"{self.port}:11434",
                "-v", "ollama_data:/root/.ollama",  # persistent volume
                "ollama/ollama:latest",
                "serve"
            ], check=True)
            print("Waiting 5 seconds for container to start...")
            time.sleep(5)

        except Exception as e:
            raise RuntimeError(f"Failed to start Ollama container: {e}")


    def _ensure_model(self, model_name: str) -> None:
        """Ensure the requested Ollama model is available in the container."""
        try:
            ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                stream=False,
            )
        except ResponseError as exc:
            if exc.status_code == 404:
                print(f"Model '{model_name}' not found. Pulling into container...")
                ollama.pull(model_name)
            else:
                raise
    
    def _stop_container(self) -> None:
        """Stop the Ollama Docker container if it is running."""
        try:
            # Check if container is running
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={self.container_name}"],
                capture_output=True,
                text=True,
            )
            container_id = result.stdout.strip()

            if container_id:
                print(f"Stopping Ollama container '{self.container_name}'...")
                subprocess.run(["docker", "stop", self.container_name], check=True)
                print("Container stopped.")
            else:
                print(f"Ollama container '{self.container_name}' is not running, nothing to stop.")

        except Exception as e:
            print(f"[Warning] Failed to stop Ollama container: {e}")

    def chat(
        self,
        question: str,
        k: int = 5,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Ask a question using RAG-enhanced context."""
        docs: list[RetrievedDocument] = self.retriever.retrieve(
            query=question,
            top_k=k,
        )

        prompt: str = self._build_rag_prompt(question, docs)
        self.history.append({"role": "user", "content": prompt})

        if stream:
            return self._stream_response()

        response = ollama.chat(
            model=self.model,
            messages=self.history,
            stream=False,
        )
        answer: str = response["message"]["content"]
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def _stream_response(self) -> Iterator[str]:
        """Stream the assistant response token-by-token."""
        response_text: str = ""

        for chunk in ollama.chat(
            model=self.model,
            messages=self.history,
            stream=True,
        ):
            partial: str = chunk["message"]["content"]
            response_text += partial
            yield partial

        self.history.append({"role": "assistant", "content": response_text})

    def _build_rag_prompt(
        self,
        question: str,
        docs: list[RetrievedDocument],
    ) -> str:
        """Construct a RAG prompt by injecting retrieved context."""
        context: str = "\n\n".join(doc["document"] for doc in docs)
        return f"Context:\n{context}\n\nQuestion:\n{question}"
