# MLOps End-to-End: Previsão de Preços de Imóveis

Um projeto MLOps abrangente para previsão de preços de imóveis usando Random Forest, apresentando pipeline completo de CI/CD, monitoramento e automação de implantação.

##  Visão Geral da Arquitetura

Este projeto implementa um pipeline MLOps completo com:

- **Gerenciamento de Dados**: Carregamento, pré-processamento e validação automatizados de dados
- **Treinamento de Modelos**: Random Forest com otimização de hiperparâmetros
- **Rastreamento de Experimentos**: Integração com MLflow para versionamento de modelos
- **Serviço de API**: FastAPI para previsões em tempo real
- **Containerização**: Configuração do Docker e Docker Compose
- **Orquestração**: Configurações de implantação do Kubernetes
- **Automação de Pipelines**: Pipelines do Kubeflow para fluxos de trabalho de ML
- **Monitoramento**: Detecção de desvios de dados e rastreamento de desempenho do modelo

##  Início Rápido

### Pré-requisitos
- Python 3.9+
- Docker e Docker Compose
- Kubernetes (opcional)
- Make (opcional, para comandos de conveniência)

### 1. Configurar Ambiente
```bash
# Clonar e configurar
git clone https://github.com/SauloHenriqueAguiar/MLOpsEndtoEnd
cd MLOpsEndtoEnd

# Usando Make (recomendado)
make setup

# Ou manualmente
pip install -r requirements.txt
pip install -e .
mkdir -p data/{raw,processed,models} logs artifacts
cp .env.example .env
```

### 2. Gerar Dados e Treinar Modelo
```bash
# Gerar dados sintéticos
make data
# ou: python scripts/generate_data.py --samples 1000

# Treinar modelo com otimização de hiperparâmetros
make train
# ou: python scripts/train_model.py --optimize
```

### 3. Iniciar Serviço da API
```bash
# Iniciar API localmente
make api
# ou: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Testar API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "area": 120.5,
    "quartos": 3,
    "banheiros": 2,
    "idade": 5.0,
    "garagem": 1,
    "bairro": "Zona Sul"
  }'
```

### 4. Stack Completa com Docker
```bash
# Iniciar stack MLOps completa
make docker-up
# ou: docker-compose -f docker/docker_compose.yml up -d

# Acessar serviços:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Jupyter: http://localhost:8888
```

##  Estrutura do Projeto

```
MLOpsEndtoEnd/
├── configs/                 # Arquivos de configuração
│   ├── config.yaml         # Configuração principal
│   └── model_config.yaml   # Configuração específica do modelo
├── data/                   # Armazenamento de dados
│   ├── raw/               # Dados brutos
│   ├── processed/         # Dados processados
│   └── models/            # Modelos treinados
├── docker/                # Configurações Docker
│   ├── Dockerfile.api     # Container da API
│   ├── Dockerfile.training # Container de treinamento
│   └── docker_compose.yml # Stack completa
├── kubernetes/            # Configurações de implantação K8s
├── kubeflow/             # Pipelines Kubeflow
├── notebooks/            # Notebooks Jupyter
├── src/                  # Código fonte
│   ├── api/              # Aplicação FastAPI
│   ├── data/             # Processamento de dados
│   ├── features/         # Engenharia de features
│   ├── models/           # Modelos ML
│   ├── monitoring/       # Componentes de monitoramento
│   └── utils/            # Utilitários
├── scripts/              # Scripts utilitários
├── tests/                # Suíte de testes
└── Makefile             # Automação de build
```

##  Configuração

O projeto usa arquivos de configuração YAML:

- `configs/config.yaml`: Configuração principal do projeto
- `configs/model_config.yaml`: Parâmetros específicos do modelo
- `.env`: Variáveis de ambiente (copie de `.env.example`)

##  Testes

```bash
# Executar todos os testes
make test
# ou: pytest tests/ -v --cov=src

# Executar categorias específicas de teste
pytest tests/test_api.py -v
pytest tests/test_models.py -v
```

##  Implantação

### Implantação Docker
```bash
# Construir imagens
make docker

# Iniciar serviços
docker-compose -f docker/docker_compose.yml up -d
```

### Implantação Kubernetes
```bash
# Implantar no Kubernetes
make k8s-deploy
# ou: kubectl apply -f kubernetes/

# Verificar implantação
kubectl get pods -n mlops
```

### Pipelines Kubeflow
```bash
# Compilar e executar pipelines
cd kubeflow/
python pipeline.py
```

##  Monitoramento

O projeto inclui monitoramento abrangente:

- **Detecção de Desvio de Dados**: Detecção automática de mudanças na distribuição dos dados
- **Performance do Modelo**: Monitoramento contínuo da qualidade das previsões
- **Métricas da API**: Monitoramento de requisições/respostas com Prometheus
- **Logging**: Logging estruturado com níveis configuráveis

##  Pipeline CI/CD

O projeto suporta CI/CD automatizado com:

1. **Qualidade do Código**: Testes automatizados e linting
2. **Treinamento de Modelo**: Retreinamento automatizado em mudanças de dados
3. **Validação de Modelo**: Gates de qualidade antes da implantação
4. **Implantação**: Implantação automatizada para staging/produção

##  Integração MLflow

- **Rastreamento de Experimentos**: Todas as execuções de treinamento registradas automaticamente
- **Registro de Modelos**: Armazenamento versionado de modelos
- **Servir Modelos**: Capacidades diretas de servir modelos
- **Comparação de Métricas**: Comparação fácil de versões de modelos

##  Desenvolvimento

### Formatação de Código
```bash
make format
# ou: black src/ tests/ scripts/
```

### Adicionando Novas Funcionalidades
1. Criar branch de feature
2. Adicionar testes para nova funcionalidade
3. Atualizar configuração se necessário
4. Submeter pull request

##  Documentação da API

Uma vez que a API esteja rodando, visite:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints Principais
- `POST /predict`: Previsão única
- `POST /predict/batch`: Previsões em lote
- `GET /health`: Verificação de saúde
- `GET /model/info`: Informações do modelo
- `GET /metrics`: Métricas de performance

##  Solução de Problemas

### Problemas Comuns

1. **Modelo não encontrado**: Execute `make train` para treinar o modelo primeiro
2. **Conflitos de porta**: Verifique se as portas 8000, 5000, 8888 estão disponíveis
3. **Problemas Docker**: Certifique-se de que o daemon Docker está rodando
4. **Erros de permissão**: Verifique permissões de arquivo nos diretórios de dados

### Logs
- Logs da aplicação: `logs/app.log`
- Logs Docker: `docker-compose logs -f`
- Logs Kubernetes: `kubectl logs -f deployment/house-price-api -n mlops`

##  Contribuindo

1. Faça fork do repositório
2. Crie uma branch de feature
3. Faça suas alterações
4. Adicione testes
5. Submeta um pull request

##  Licença

Este projeto está licenciado sob a Licença MIT.

##  Tags

`MLOps` `Machine Learning` `FastAPI` `Docker` `Kubernetes` `MLflow` `Kubeflow` `Random Forest` `Previsão de Preços de Imóveis` `CI/CD`