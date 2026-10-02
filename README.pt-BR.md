# julia1-cli

*English: [README.md](README.md)*

Um runtime nativo em C++/ggml (Metal e CPU) para o [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1), o modelo
de decisão da Supersonic Labs ([página oficial](https://supersoniclabs.ia.br/pt/julia-1/)), em GGUF. Ele reproduz os
resultados do PyTorch original e vem como CLI com servidor HTTP, API C e pacote Python (`julia1-gguf`).

Os arquivos GGUF estão no Hugging Face: [andrelucas/Julia-1-GGUF](https://huggingface.co/andrelucas/Julia-1-GGUF).

Os nomes `julia1-cli`, `julia1-gguf` e `julia1` se referem ao modelo Julia-1; o projeto não tem relação com a
linguagem de programação Julia. É um port independente, sem afiliação com a Supersonic Labs nem endosso dela.

## Julia-1

O Julia-1 é um **modelo de decisão, não um modelo generativo**. Uma requisição é um `state` (texto ou JSON), uma
`question`, um `type` (`choice`, `score` ou `noul`, um booleano) e 2–20 `options`; o modelo devolve um logit por
opção, e quem chama aplica softmax e argmax. É um encoder mmBERT-small (ModernBERT: 22 camadas, hidden 384,
vocabulário de 256 000, contexto de 8192) seguido de um módulo de decisão tipado (um embedding do tipo de pergunta,
duas camadas transformer e um scorer lido no marcador `<mask>` antes de cada opção); 144,19 M de parâmetros.

Os arquivos GGUF estão no layout do llama.cpp para este modelo (arquitetura `modern-bert` com dois blocos de decisão,
gravados pelo `convert_hf_to_gguf.py` do llama.cpp), então os mesmos arquivos rodam no julia1-cli e no llama.cpp
oficial, no master ([abaixo](#julia-1-no-llamacpp)). O julia1-cli também carrega o layout `julia1` dos arquivos da
versão 0.1.0. O [SPEC.md](SPEC.md) define os dois layouts, o tokenizador, a codificação das requisições, o forward e o
protocolo de validação.

## Conteúdo

| caminho | o quê |
| --- | --- |
| `src/` | o runtime: carregador GGUF, tokenizador, codificação, grafo ggml, CLI, servidor HTTP, API C |
| `patches/ggml-julia1.patch` | mudanças no ggml (MIT): kernels Metal F32 exatos, epílogos fundidos nas GEMMs do Metal, ganhos na CPU; aplicado na compilação |
| `julia1_gguf/` | o pacote Python `julia1-gguf`: o mesmo runtime pela API C, ou um runtime em numpy puro |
| `tools/convert_julia1_to_gguf.py` | conversor do checkpoint safetensors do upstream para o layout legado `julia1` |
| `tools/eval_gguf.py`, `tools/compare_replay.py`, `tools/tokenizer_parity*.py` | validação contra os dados de referência |
| `tools/systemone_parity.py` | as mesmas 2000 perguntas tipadas por um servidor HTTP (`/v1/systemone` do llama-server ou `julia1-cli serve`) |
| `data/reference/` | logits de referência do PyTorch original: 2000 perguntas do typed-decisions e 100 casos de paridade |
| `data/tokenizer-corpus.jsonl` | 4595 strings para a verificação de paridade do tokenizador |
| `docs/server.md`, `docs/BENCHMARKS.md` | a API HTTP; configuração e tabelas completas dos benchmarks |
| `packaging/`, `tools/package_release.sh` | opcional: um arquivo autocontido para macOS e uma wheel, gerados localmente |

## Instalação

Não há binários pré-compilados publicados; compile a partir do código-fonte.

**A partir do código-fonte** (CMake ≥ 3.18, um compilador C++17, `patch`; a configuração baixa o release v0.5.0 do
llama.cpp, com SHA-256 fixado, e aplica `patches/ggml-julia1.patch` a uma cópia do ggml dele):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8          # build/julia1-cli and build/libjulia1_gguf_native.dylib (.so on Linux)
```

* Offline, a partir de um llama.cpp v0.5.0 descompactado: acrescente
  `-DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<caminho absoluto>`.
* `-DJULIA1_GGML_PATCH=OFF` compila contra o ggml sem modificações: os resultados continuam exatos, e o `--precise` no
  Metal fica cerca de 4x mais lento.
* macOS: Metal, e Accelerate para o dispositivo CPU. Linux: só CPU (sgemm do llamafile; `-DJULIA1_BLAS=ON
  -DGGML_BLAS_VENDOR=OpenBLAS` para OpenBLAS). O build para Linux é experimental: a CI o compila (com falha
  permitida), e ele não foi validado contra os dados de referência. Sem CUDA.

**Pacote Python a partir do código-fonte** (compila a biblioteca nativa com o CMake, como acima; numpy e gguf são as
únicas dependências):

```sh
pip install .              # or: pip install ".[fast]" for the HF tokenizers used by the numpy backend
```

## Início rápido

Baixe o `Julia-1-F32.gguf` de [andrelucas/Julia-1-GGUF](https://huggingface.co/andrelucas/Julia-1-GGUF) e rode o
exemplo do README do upstream:

```sh
cat > questions.jsonl <<'JSONL'
{"state": "I was charged twice for the same order.", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery", "access": "Account access and login"}}}}
JSONL
build/julia1-cli --model Julia-1-F32.gguf predict --input questions.jsonl --output answers.jsonl --max-length 8192 --head-length 512 --strict
cat answers.jsonl
```

```json
{"answers":{"team":{"type":"choice","probabilities":{"billing":0.8544651362474033,"shipping":0.14277066945266953,"access":0.002764194299927172},"choice":"billing","max_probability":0.8544651362474033}}}
```

(Apple M1 Pro, Metal. Na CPU a mesma requisição dá billing 0.8544669661607737: F32 exato, com outra ordem de soma.)

## CLI

```
julia1-cli --model <gguf> [--device auto|cpu|metal] [--threads N] [--precise|--fast] [--batch B] [--timing] <command> --input <in.jsonl> --output <out.jsonl>
  decide   [--max-length 8192] [--head-length 256] [--strict]   {"state","question","options","type"} -> {"logits","index","probabilities","ids","markers","qtype"}
  predict  [--max-length 8192] [--head-length 256] [--strict]   {"state","questions":{...}} -> {"answers":{...}} (upstream engine.predict)
  replay   {"id","ids","markers","qtype"} -> {"id","logits"}     (an already-encoded sequence)
  tokenize {"text"} -> {"ids"}
julia1-cli --model <gguf> [...] serve [--host 127.0.0.1] [--port 8080] [--api-key KEY] [--devices metal,cpu]
```

* `--device auto` (padrão) usa o Metal quando ele inicializa, senão a CPU. `--threads` (padrão 4) deve ficar abaixo
  do número de núcleos de desempenho livres.
* `--precise` (padrão) roda kernels F32 exatos no Metal; `--fast` usa operandos em meia precisão (mais rápido; no
  arquivo F16, logits a menos de 0,6 da referência e as mesmas 2000/2000 decisões do conjunto typed). A CPU é sempre
  exata.
* `--batch B` agrupa B linhas por comprimento em grafos com padding, para throughput.
* `decide` e `predict` usam por padrão a codificação do `julia.load_model` do upstream (`max_length` 8192,
  `head_length` 256, não estrito). A política recomendada pelo upstream é 8192 / 512 / estrito; o protocolo de
  avaliação é 1024 / 512 (typed) ou 256 (paridade) / estrito.

## Servidor HTTP

`julia1-cli serve` mantém o modelo carregado e agrupa requisições concorrentes dinamicamente
([docs/server.md](docs/server.md), em inglês):

| método, caminho | corpo | resposta |
| --- | --- | --- |
| `POST /v1/predict` | `{"state", "questions"}` (`engine.predict` do upstream), o legado `{"rows": [...]}`, ou um array deles | `{"answers": {...}}` ou `{"predictions": [...]}` |
| `POST /v1/decide` | `{"rows": [{state, question, options, type}]}` | `{"results": [{"index", "probabilities", "logits"}]}` |
| `POST /v1/logits` | `{"rows": [...]}` | `{"logits": [[...]]}` |
| `POST /v1/tokenize` | `{"texts": [string]}` | `{"ids": [[...]]}` |
| `GET /health`, `GET /v1/model`, `GET /` | - | status, informações do modelo, uma página de playground |

```sh
build/julia1-cli --model Julia-1-F32.gguf serve            # http://127.0.0.1:8080
curl -s http://127.0.0.1:8080/v1/predict -H 'Content-Type: application/json' -d @questions.jsonl
```

Segurança: o servidor escuta em `127.0.0.1` por padrão e não tem TLS nem limite de taxa. Defina `JULIA1_API_KEY` (ou
`--api-key`) para exigir `Authorization: Bearer <chave>` em `/v1/*`, e coloque um proxy reverso na frente antes de
escutar em outro endereço.

## Pacote Python

```python
from julia1_gguf import load_model

engine = load_model("Julia-1-F32.gguf", max_length=8192, head_length=512, strict_encoding=True)
result = engine.predict(
    state="I was charged twice for the same order.",
    questions={"team": {"type": "choice", "instructions": "Which team should handle this request?",
                        "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery",
                                     "access": "Account access and login"}}},
)
print(result["answers"]["team"]["choice"])  # billing
```

`load_model(path, max_length=None, head_length=256, strict_encoding=False, *, backend="auto", device="auto",
precise=True, threads=4, batch=1)` tem os padrões do upstream. `backend="native"` roda o runtime do julia1-cli pela API
C (o padrão quando a biblioteca está instalada), `backend="numpy"` um runtime em numpy puro. Os dois oferecem
`predict(state=..., questions=...)`, o legado `predict(rows)`, `logits(rows)`, `replay(ids, markers, qtype)`,
`encode(row)` e `tokenize(text)`. `JULIA1_GGUF_LIB` escolhe outra biblioteca nativa, e `JULIA1_GGUF_VERBOSE=1` mostra
o log completo do ggml. `python -m julia1_gguf` (script `julia1-gguf`) é uma CLI para o runtime numpy com o mesmo
contrato do julia1-cli.

## API C

`julia1_gguf.h` (biblioteca `libjulia1_gguf_native`): JSON na entrada, JSON na saída, os mesmos resultados do
julia1-cli.

```c
void * engine = julia1_engine_new("{\"model\": \"Julia-1-F32.gguf\", \"head_length\": 512, \"strict\": true}", &error);
char * result = julia1_engine_call(engine, "predict_typed", "{\"state\": ..., \"questions\": {...}}", &error);
julia1_string_free(result);
julia1_engine_free(engine);
```

Métodos: `info`, `logits`, `predict_rows`, `predict_typed`, `replay`, `encode`, `tokenize`. Um exemplo completo está
no [guia rápido](packaging/README-quickstart.md) (em inglês).

## Arquivos do modelo

Em [andrelucas/Julia-1-GGUF](https://huggingface.co/andrelucas/Julia-1-GGUF), convertidos com o
`convert_hf_to_gguf.py` do llama.cpp ([PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818)); os mesmos
arquivos rodam no llama.cpp:

| arquivo | tamanho | uso |
| --- | --- | --- |
| `Julia-1-F32.gguf` | 592 MB | referência exata |
| `Julia-1-F16.gguf` | 303 MB | metade do tamanho; kernels exatos ou `--fast` |

Os arquivos no layout `julia1` do julia1-cli 0.1.0 (a revisão anterior do repositório) continuam carregando; os
arquivos só do encoder, `laya` e `F16-embdQ8_0` daquela versão não são mais publicados. O julia1-cli para com um erro
em arquivos só do encoder e em modelos da mesma família com outro template de prompt (Laya). Matrizes em BF16 e Q8_0
mudam decisões (as conversões da ggml-org, [ggml-org/Julia-1-GGUF](https://huggingface.co/ggml-org/Julia-1-GGUF):
1986 e 1946 de 2000 no julia1-cli) e não são publicadas. Para converter o checkpoint do upstream você mesmo, com o
llama.cpp master:

```sh
python convert_hf_to_gguf.py <SupersonicLabs/Julia-1 download> --outtype f32 --outfile Julia-1-F32.gguf   # or f16
```

O `tools/convert_julia1_to_gguf.py` (Python com `numpy`, `safetensors` e `gguf`) continua gravando o layout legado
`julia1`, com tipos por tensor ([SPEC.md](SPEC.md) §2.3).

## Resultados

Apple M1 Pro (8 núcleos de desempenho + 2 de eficiência, 16 GB), macOS 27. Referência: o PyTorch do upstream na CPU em
FP32, que reproduz o resultado publicado pelo upstream no conjunto de teste typed-decisions (acurácia 1451/2000).
Conjunto typed = as 2000 perguntas dele (`max_length` 1024, `head_length` 512, estrito); latência = mediana de uma
requisição por chamada com a entrada já tokenizada (de ponta a ponta soma 0,2–0,3 ms); lote de 16 = requisições/s
pelo batching próprio de cada runtime.

| runtime | ms por requisição | × PyTorch CPU | requisições/s, lote de 16 | decisões iguais à referência | acurácia | máx. abs. do logit |
| --- | --- | --- | --- | --- | --- | --- |
| julia1-cli, Metal `--fast`, arquivo F16 | 11,4 | 5,3× | 111 | 2000/2000 | 1451 | 0,56 |
| julia1-cli, Metal exato (padrão), arquivo F32 | 12,3 | 4,9× | 102 | 2000/2000 | 1451 | 8,2e-4 |
| julia1-cli, CPU, arquivo F32, 4 threads | 41,9 | 1,43× | 32 (6 threads) | 2000/2000 | 1451 | 8,4e-4 |
| MLX fp16 (julia-mlx) | 15,6 | 3,85× | 87 | 1997/2000 | 1452 | 0,48 |
| MLX fp32 (julia-mlx) | 17,2 | 3,50× | 77 | 2000/2000 | 1451 | 8,6e-4 |
| PyTorch MPS (código do upstream) | 30,0 | 2,01× | 62 | 2000/2000 | 1451 | 5,0e-4 |
| **PyTorch CPU, 4 threads (upstream)** | 60,1 | 1,00× | 26 | 2000/2000 | 1451 | 2,3e-4 |
| ONNX Runtime CPU, 8 threads (Julia-1-ONNX) | 85,3 | 0,70× | 11 | 2000/2000 | 1451 | 7,6e-4 |

* Medido com o julia1-cli 0.1.0 nos arquivos dele, no layout `julia1`; com os mesmos pesos F32, a versão 0.2.0 dá
  saída idêntica byte a byte a partir do `Julia-1-F32.gguf` no layout do llama.cpp.
* Os 100 casos de paridade dão 100/100 decisões em todas as configurações do julia1-cli acima; o backend nativo do
  pacote Python dá os mesmos logits do julia1-cli e soma 0,0–0,3 ms por chamada.
* De ponta a ponta, o tokenizador e a codificação do julia1-cli produzem exatamente os ids de referência (0
  divergências nas 2000 requisições typed e nas 100 de paridade); o tokenizador coincide com o `tokenizers` da HF nas
  4595 strings de `data/tokenizer-corpus.jsonl`, nos dois layouts.
* O `julia1-cli serve` responde 103 requisições/s (exato) e 112 (fast) com 32 clientes simultâneos, 128 e 136 com
  `--devices metal,cpu`. Do início do processo à primeira resposta são cerca de 0,2 s.

Configuração, protocolo e as tabelas completas (por comprimento de sequência, de ponta a ponta, servidor, partida a
frio): [docs/BENCHMARKS.md](docs/BENCHMARKS.md) (em inglês). Para verificar um build contra os dados de referência:

```sh
python tools/eval_gguf.py --runtime cli --cli build/julia1-cli --model Julia-1-F32.gguf --suite typed --mode e2e \
    --min-argmax 2000 --max-abs 1e-3 --accuracy 1451
python tools/eval_gguf.py --runtime cli --cli build/julia1-cli --model Julia-1-F32.gguf --suite parity --mode replay \
    --device cpu --min-argmax 100 --max-abs 2e-3
```

## Julia-1 no llama.cpp

O llama.cpp master roda o Julia-1 a partir dos mesmos arquivos desde 2026-10-02
([PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818); nenhum release com tag o inclui ainda). O
`llama-server` recebe em `POST /v1/systemone` o mesmo corpo `{"state", "questions"}` do `/v1/predict` do
`julia1-cli serve`:

```sh
llama-server -m Julia-1-F16.gguf -b 2048 -ub 2048      # -ub >= the longest prompt in tokens (up to 8192)
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d @questions.jsonl
```

* O `-ub` (padrão 512) precisa comportar o prompt inteiro: com os padrões, 35 das 2000 perguntas typed falham com HTTP
  500 (`input (N tokens) is too large to process`).
* Envie um state JSON como string serializada por você (`json.dumps` do Python). O `tojson` do llama.cpp escreve
  floats com 6 dígitos significativos e descarta o `.0` (`532.0` → `532`, `1234567.5` → `1.23457e+06`): com o state
  enviado como objeto, 480 dos 2000 prompts typed são tokenizados de forma diferente da do upstream, e 51 das 52
  decisões alteradas vêm deles.
* O llama.cpp roda o `gelu` do ModernBERT como a GeGLU do ggml com a aproximação por tanh, enquanto o upstream usa a
  erf exata; com o state enviado como string, essa é a principal fonte de erro que resta. Na CPU, acrescente `-fa off`
  (o flash attention soma um pouco de erro ali).

As 2000 perguntas typed, uma pergunta por requisição, HTTP sequencial a partir do mesmo cliente para os dois
servidores (Apple M1 Pro, llama.cpp master `1fb7ef3e`). As diferenças de logit são recuperadas das probabilidades
devolvidas; os ms incluem a ida e volta HTTP, então não são comparáveis com a tabela acima:

| servidor, arquivo, dispositivo | decisões iguais à referência | mediana / máx. abs. da diferença de logit | ms por requisição |
| --- | --- | --- | --- |
| llama-server, F32, Metal | 1999/2000 | 0,037 / 0,40 | 22,8 |
| llama-server, F32, CPU, 6 threads | 1999/2000 | 0,030 / 0,29 | 62,2 |
| llama-server, F16, Metal | 2000/2000 | 0,039 / 0,58 | 21,8 |
| julia1-cli serve, F32, Metal exato | 2000/2000 | 5,0e-5 / 8,5e-4 | 13,4 |
| julia1-cli serve, F16, Metal `--fast` | 2000/2000 | 0,023 / 0,41 | 12,5 |

O llama-server rodou com `-b 1024 -ub 1024` e o state enviado como string; o julia1-cli, com os arquivos da versão
0.1.0, no layout `julia1`. O julia1-cli é cerca de 1,7× mais rápido aqui e exato no Metal. Os arquivos BF16 e Q8_0 da
ggml-org dão 1987 e 1947 de 2000 no llama-server. Detalhes e as outras configurações:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) §F (em inglês).

## Limitações

* O Julia-1 compara as opções que recebe; não é um modelo de conhecimento nem de raciocínio. Avalie-o nas suas
  próprias perguntas e opções antes de agir com base na resposta.
* Testado só em macOS em Apple silicon. O Linux compila a partir do código-fonte e é experimental; sem CUDA.
* O ggml compila os shaders Metal quando um processo inicia; quando o cache de shaders do macOS falha (na primeira
  execução numa máquina, ou depois que outro build do ggml rodou), a partida leva cerca de 20 s em vez de 0,1–0,3 s.
  `--device cpu` também é afetado, porque todos os backends são registrados na partida.
* Um build contra o ggml sem modificações (`-DJULIA1_GGML_PATCH=OFF`) é exato, mas cerca de 4x mais lento no modo
  `--precise` no Metal.
* O batching aplica padding às requisições agrupadas: os logits ficam dentro dos limites, mas as probabilidades podem
  diferir das do lote de 1 nos últimos dígitos.

## Referências

* [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1): o modelo, os pesos e o código de
  referência da Supersonic Labs (Apache-2.0); página oficial: https://supersoniclabs.ia.br/pt/julia-1/.
* [SupersonicLabs/Julia-1-ONNX](https://huggingface.co/SupersonicLabs/Julia-1-ONNX): a exportação ONNX da Supersonic
  Labs; os 100 casos de paridade dela fazem parte de `data/reference/`.
* [zainmerchan/Julia-1-MLX](https://huggingface.co/zainmerchan/Julia-1-MLX) e o runtime
  [julia-mlx](https://github.com/zm2231/julia-mlx) (MIT): o port para MLX, usado como referência de velocidade e
  acurácia.
* [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) (MIT; arXiv:2509.06888): o encoder e o
  tokenizador sobre os quais o Julia-1 foi construído. ModernBERT, da Answer.AI e da LightOn (arXiv:2412.13663): a
  arquitetura do encoder.
* [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) v0.5.0 (MIT): a biblioteca de tensores, o formato GGUF e
  o gguf-py; o sgemm do llamafile (MIT), o [cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT) e o
  [nlohmann/json](https://github.com/nlohmann/json) (MIT) vêm com ele.
* [PR #29818 do llama.cpp](https://github.com/ggml-org/llama.cpp/pull/29818) (MIT): o Julia-1 no llama.cpp
  (`/v1/systemone` do `llama-server`) e o `convert_hf_to_gguf.py` que gravou os arquivos publicados;
  [ggml-org/Julia-1-GGUF](https://huggingface.co/ggml-org/Julia-1-GGUF): as conversões BF16 e Q8_0 da ggml-org, usadas
  para comparação.
* [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (Apache-2.0): o conjunto de
  teste dos dados de referência.
* [PyTorch](https://pytorch.org/), [transformers](https://github.com/huggingface/transformers) e
  [tokenizers](https://github.com/huggingface/tokenizers) da Hugging Face, [ONNX Runtime](https://onnxruntime.ai/) e
  [MLX](https://github.com/ml-explore/mlx): usados só para validação e benchmarks.

## Licença

Apache-2.0 ([LICENSE](LICENSE)), exceto `patches/ggml-julia1.patch`, que é MIT como o ggml. As atribuições estão no
[NOTICE](NOTICE), e as licenças do código de terceiros compilado nos binários, no
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES). Os pesos do modelo são Apache-2.0, da Supersonic Labs.
