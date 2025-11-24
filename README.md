Sistema de Recomendação de Filmes

Um sistema completo de recomendação utilizando Filtragem Baseada em Conteúdo, Filtragem Colaborativa Item-Item, interface em Streamlit, backend em FastAPI, e um Agente Cinéfilo para interagir com os dados do sistema.

EQUIPE:
Caio Jorge da Cunha Queiroz — 2315310028
Lucas Maciel Gomes — 2315310014
Izabella de Lima Catrinck — 2315310033

Principais Funcionalidades
1. Recomendações Baseadas em Conteúdo (Content-Based)
  Utiliza TF-IDF aplicado às informações textuais dos filmes (gêneros e sinopse).
  Calcula a similaridade por cosseno entre o perfil do usuário e os filmes do catálogo.

2. Filtragem Colaborativa Item-Item
  Recomenda filmes com base em padrões de comportamento entre usuários.
  Mede similaridade entre filmes avaliados por usuários semelhantes.

3. Catálogo de Filmes
  Lista de filmes para explorar.
  Possibilidade de avaliar filmes para melhorar o perfil.

4. Agente Cinéfilo (Chat com IA)
  O usuário pode fazer perguntas sobre o sistema, dados, métricas e recomendações.
  Utiliza API Groq (necessário definir a chave em .env).

5. Autenticação
  Login e cadastro de usuários.
  Preferências salvas individualmente.

Tecnologias Utilizadas
  FastAPI (endpoints REST para recomendações e dados)
  Scikit-learn (TF-IDF, Similaridade do Cosseno)
  Pandas (tratamento de dados)
  NumPy
  Streamlit
  HTML/CSS básico para componentes customizados

Infraestrutura
Autenticação simples por sessão
Agente de IA usando Groq API
uv / pip para gerenciamento de dependências

Estrutura dos Dados
1. ratings_final_df.csv (avaliações dos usuários)
Coluna	Descrição
userId	Identificador único do usuário
movieId	Identificador único do filme
rating	Nota do filme (1 a 5) atribuída pelo usuário
Estatísticas

👥 Usuários únicos: 317

🎬 Filmes avaliados: 356

2. movies_final_df.csv (catálogo de filmes)
Coluna	Descrição
id	ID do filme (equivalente a movieId)
title	Título original
genres	Gêneros brutos
genres_clean	Gêneros tratados (lista limpa)
overview	Sinopse
production_companies	Estúdios
companies_text	Estúdios tratados em texto
tagline	Frase de impacto
popularity	Métrica de popularidade
release_date	Data de lançamento
vote_average	Média de votos
vote_count	Número de votos
content	Campo final concatenado (gêneros + sinopse + tags), utilizado no TF-IDF

* Pipeline do Sistema
1. Pré-processamento

Remoção de stopwords

Normalização do texto

Construção da coluna content com:

gêneros + overview + tagline + studios + outras features textuais

2. Vetorização

Modelo: TF-IDF (Term Frequency–Inverse Document Frequency)

Hyperparams comuns:

ngram_range = (1,2)
max_features = 5000
stop_words = 'english'

3. Perfil do Usuário

Perfil = média dos vetores TF-IDF dos filmes avaliados positivamente.

4. Cálculo da Similaridade

Métrica: Cosine Similarity

Retorna top-K filmes mais similares ao perfil.

5. Filtragem Colaborativa

Similaridade entre itens por correlação de notas.

Recomendação baseada em filmes "vizinhos" ao já avaliado.

Métricas de Avaliação

Usamos Precision, Recall e F1-Score para medir a qualidade das recomendações.

🔹 Precision (Precisão)

Pergunta:

Das recomendações feitas, quantas estavam corretas?

Cálculo:

Precision = acertos / número de recomendações


Interpretação:

Alta precision → o sistema recomenda poucos filmes ruins.

Normal em sistemas por conteúdo: 0.50–0.70.

🔹 Recall (Revocação)

Pergunta:

De todos os filmes relevantes para o usuário, quantos foram recomendados?

Cálculo:

Recall = acertos / total de filmes relevantes


Interpretação:

Alta recall → boa cobertura dos gostos do usuário.

Normal em TF-IDF: 0.30–0.50.

🔹 F1-Score

Pergunta:

O sistema está equilibrado entre recomendar certo e encontrar tudo que o usuário gosta?

Cálculo:

F1 = 2 * (Precision * Recall) / (Precision + Recall)


Interpretação:

Bom quando está entre 0.40 e 0.55

Excelente se > 0.60

Como Executar o Sistema
1. Instalar uv
pip install uv

2. Instalar dependências
uv sync

3. Iniciar o backend (FastAPI)
uvicorn backend.main:app --reload

4. Rodar o Frontend (Streamlit)
streamlit run frontend/app.py

⚠️ Atenção: Agente Cinéfilo

Para usar o agente de IA:

Crie um arquivo .env na raiz

Adicione:

GROQ_API_KEY=sua_chave_aqui

