"""
Gerador de corpus sintético de e-commerce.

Produz um catálogo realista de produtos com descrições longas (200-500
palavras) e planta, de forma controlada, os três fenômenos linguísticos que o
benchmark precisa medir:

* **Polissemia** — ``mouse`` (periférico) x ``mouse``/``rato`` (brinquedo de
  pet shop); ``teclado`` (informática) x ``teclado`` (instrumento musical);
  ``tablet`` (dispositivo) x ``tablete`` (chocolate); ``bateria``, ``manga``,
  ``vela``, ``placa``.
* **Sinonímia** — ``sem fio``/``wireless``, ``notebook``/``laptop``,
  ``fone``/``headset``, ``celular``/``smartphone``, ``periférico``/``acessório``.
* **Termos raros** — atributos de baixa frequência como ``bluetooth 5.0``,
  ``hot-swappable``, ``16000 DPI``, ``hall effect``.

Além de ``data/products.json``, o módulo grava ``data/qrels.json`` com
julgamentos de relevância graduados (0-3) para as cinco consultas da análise
qualitativa, derivados dos atributos internos usados na geração. Isso permite
medir P@5 e nDCG@5 objetivamente, em vez de julgar as SERPs no olho.

Uso:
    python src/corpus_generator.py [--size 5000] [--output data/products.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Blocos de vocabulário
# --------------------------------------------------------------------------- #

BRANDS: Tuple[str, ...] = (
    "Nexora", "Volttix", "Kyroshi", "Bramante", "Lumora", "Ferravox",
    "Aurelith", "Tecnovia", "Zephyros", "Marlow", "Cendra", "Ipanor",
    "Vantek", "Orbelle", "Sunnara", "Petrix", "Corvale", "Amarante",
)

COLORS: Tuple[str, ...] = (
    "preto", "grafite", "branco", "cinza-chumbo", "azul-marinho", "vermelho",
    "verde-musgo", "bege", "rosa-quartzo", "prata",
)

QUALITY_ADJECTIVES: Tuple[str, ...] = (
    "robusto", "elegante", "compacto", "versátil", "resistente", "leve",
    "sofisticado", "prático", "durável", "confortável",
)

GENERIC_CLOSERS: Tuple[str, ...] = (
    "O produto acompanha manual de instruções em português e certificado de "
    "garantia do fabricante.",
    "Enviamos em embalagem reforçada, com proteção extra contra impactos "
    "durante o transporte.",
    "Todos os itens passam por inspeção de qualidade antes da expedição do "
    "centro de distribuição.",
    "A garantia legal de noventa dias é somada à garantia contratual oferecida "
    "pelo fabricante.",
    "Em caso de dúvida, nossa equipe de atendimento responde em até um dia "
    "útil pelos canais oficiais.",
    "Consulte a política de trocas e devoluções antes de finalizar a compra.",
    "As imagens são meramente ilustrativas e podem variar conforme o lote de "
    "fabricação.",
    "Disponibilizamos nota fiscal eletrônica para pessoa física e jurídica.",
    "O prazo de entrega varia conforme a região e é calculado no carrinho.",
    "Peças de reposição e acessórios complementares estão disponíveis em "
    "nossa loja oficial.",
)

FILLER_SENTENCES: Tuple[str, ...] = (
    "O acabamento recebeu tratamento antirrisco que preserva a aparência "
    "original mesmo após meses de uso intenso.",
    "O projeto foi validado em testes de laboratório que simulam anos de "
    "utilização cotidiana.",
    "A relação entre custo e benefício coloca este item entre os mais "
    "procurados de sua faixa de preço.",
    "Clientes destacam a facilidade de instalação e a clareza das instruções "
    "que acompanham o produto.",
    "O design discreto combina com ambientes residenciais e corporativos sem "
    "destoar da decoração.",
    "Cada detalhe foi pensado para reduzir o esforço do usuário durante "
    "jornadas prolongadas.",
    "O controle de qualidade rastreia número de série por lote, o que agiliza "
    "eventuais acionamentos de garantia.",
    "A linha recebeu atualizações a partir do retorno enviado por compradores "
    "nas avaliações da loja.",
    "Materiais de origem certificada reduzem o impacto ambiental do processo "
    "produtivo.",
    "O item é compatível com os principais acessórios do mesmo ecossistema da "
    "marca.",
)


# --------------------------------------------------------------------------- #
# Atributos: frases que realizam cada atributo no texto
# --------------------------------------------------------------------------- #

ATTRIBUTE_PHRASES: Dict[str, Tuple[str, ...]] = {
    # --- atributos de mouse / periféricos ---
    "optico": (
        "O sensor óptico de alta precisão rastreia o movimento com fidelidade "
        "em praticamente qualquer superfície.",
        "A leitura óptica dispensa o antigo sistema de esfera e elimina "
        "travamentos por acúmulo de sujeira.",
        "O sistema óptico mantém o cursor estável mesmo em mesas de vidro "
        "fosco ou madeira clara.",
    ),
    "sem_fio": (
        "A conexão sem fio de 2.4 GHz usa um receptor nano USB que fica quase "
        "invisível na porta do computador.",
        "A tecnologia wireless elimina cabos sobre a mesa e amplia a "
        "liberdade de movimento em até dez metros.",
        "Por ser um modelo sem fio, o produto mantém a área de trabalho livre "
        "de emaranhados.",
    ),
    "bluetooth5": (
        "O padrão Bluetooth 5.0 garante pareamento rápido, alcance ampliado e "
        "consumo de energia reduzido.",
        "Com Bluetooth 5.0 é possível alternar entre três dispositivos "
        "pareados apertando um único botão.",
        "A interface Bluetooth 5.0 dispensa receptor externo em notebooks e "
        "tablets recentes.",
    ),
    "ergonomico": (
        "O formato ergonômico acompanha a curvatura natural da mão e reduz a "
        "tensão no punho ao longo do expediente.",
        "O desenho ergonômico foi desenvolvido com apoio de fisioterapeutas "
        "para prevenir lesões por esforço repetitivo.",
        "A inclinação ergonômica de trinta graus mantém o antebraço em "
        "posição neutra durante o uso prolongado.",
    ),
    "gamer": (
        "A linha gamer entrega taxa de resposta agressiva para partidas "
        "competitivas de FPS e MOBA.",
        "Iluminação gamer personalizável e macros programáveis completam o "
        "setup de jogos.",
        "O modo gamer prioriza latência mínima em detrimento da economia de "
        "energia.",
    ),
    "recarregavel": (
        "A bateria interna recarregável rende semanas de uso e é reabastecida "
        "por cabo USB-C em duas horas.",
        "O sistema recarregável dispensa pilhas descartáveis e reduz o custo "
        "de manutenção ao longo do tempo.",
    ),
    "silencioso": (
        "Os microswitches silenciosos reduzem em até noventa por cento o "
        "ruído do clique, ideal para escritórios compartilhados.",
        "O acionamento silencioso não incomoda em reuniões e chamadas de "
        "vídeo.",
    ),
    "alta_dpi": (
        "A resolução chega a 16000 DPI, ajustável em cinco níveis por um "
        "botão dedicado.",
        "O ajuste de DPI permite alternar entre precisão cirúrgica em edição "
        "de imagem e deslocamentos rápidos em planilhas.",
    ),
    "usb_c": (
        "A conexão USB-C reversível encaixa dos dois lados e transmite dados "
        "com estabilidade.",
        "O cabo USB-C destacável facilita o transporte e a substituição em "
        "caso de desgaste.",
    ),
    # --- atributos de teclado ---
    "mecanico": (
        "O teclado mecânico usa switches individuais que oferecem resposta "
        "tátil consistente a cada tecla.",
        "A construção mecânica suporta cinquenta milhões de acionamentos por "
        "switch, muito acima das membranas convencionais.",
        "O acionamento mecânico produz curso de quatro milímetros com ponto "
        "de atuação bem definido.",
    ),
    "rgb": (
        "A iluminação RGB por tecla oferece dezesseis milhões de cores e "
        "efeitos sincronizados por software.",
        "O sistema RGB pode ser desativado por completo quando o ambiente "
        "pede discrição.",
        "Perfis de iluminação RGB ficam salvos na memória interna e "
        "acompanham o equipamento em qualquer máquina.",
    ),
    "hot_swap": (
        "A base hot-swappable permite trocar switches sem solda, apenas "
        "puxando a peça com o extrator incluso.",
    ),
    "abnt2": (
        "O layout ABNT2 traz a tecla cedilha dedicada e acentuação completa "
        "do português brasileiro.",
    ),
    "compacto": (
        "O formato compacto de setenta e cinco por cento elimina o teclado "
        "numérico e libera espaço para o movimento do mouse.",
    ),
    # --- atributos gerais ---
    "impermeavel": (
        "O tratamento impermeável protege contra respingos e chuva leve sem "
        "comprometer a respirabilidade.",
    ),
    "dobravel": (
        "A estrutura dobrável reduz o volume pela metade e cabe em armários "
        "estreitos ou no porta-malas.",
    ),
    "antiderrapante": (
        "A base antiderrapante mantém o produto firme no lugar mesmo em "
        "superfícies lisas.",
    ),
    "automatico": (
        "O modo automático ajusta a intensidade conforme a leitura dos "
        "sensores internos, sem intervenção do usuário.",
    ),
    "portatil": (
        "O peso reduzido e as dimensões enxutas tornam o item genuinamente "
        "portátil para quem vive em trânsito.",
    ),
}

ATTRIBUTE_NAME_FRAGMENT: Dict[str, str] = {
    "optico": "Óptico",
    "sem_fio": "Sem Fio",
    "bluetooth5": "Bluetooth 5.0",
    "ergonomico": "Ergonômico",
    "gamer": "Gamer",
    "recarregavel": "Recarregável",
    "silencioso": "Silencioso",
    "alta_dpi": "16000 DPI",
    "usb_c": "USB-C",
    "mecanico": "Mecânico",
    "rgb": "RGB",
    "hot_swap": "Hot-Swappable",
    "abnt2": "ABNT2",
    "compacto": "Compacto",
    "impermeavel": "Impermeável",
    "dobravel": "Dobrável",
    "antiderrapante": "Antiderrapante",
    "automatico": "Automático",
    "portatil": "Portátil",
}


# --------------------------------------------------------------------------- #
# Tipos de produto
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProductType:
    """
    Molde de uma família de produtos do catálogo.

    Attributes:
        key: Identificador canônico interno (ex.: ``"mouse"``). Não aparece no
            JSON final, mas é a base dos julgamentos de relevância.
        category: Categoria exibida ao usuário.
        base_names: Núcleos de nome sorteados na composição do título.
        attributes: Atributos opcionais que podem ser sorteados para o item.
        specs: Frases de especificação técnica específicas da família.
        uses: Frases de cenário de uso.
        price_range: Intervalo ``(mínimo, máximo)`` de preço em reais.
        weight: Peso relativo na amostragem do catálogo.
        peripheral: Se o item é um periférico de computador.
        notebook_ready: Se o item é tipicamente usado junto a um notebook.
    """

    key: str
    category: str
    base_names: Tuple[str, ...]
    attributes: Tuple[str, ...] = ()
    specs: Tuple[str, ...] = ()
    uses: Tuple[str, ...] = ()
    price_range: Tuple[float, float] = (30.0, 500.0)
    weight: float = 1.0
    peripheral: bool = False
    notebook_ready: bool = False
    extra_terms: Tuple[str, ...] = field(default=())


PRODUCT_TYPES: Tuple[ProductType, ...] = (
    # ----------------------- Informática / periféricos --------------------- #
    ProductType(
        key="mouse",
        category="Informática",
        # Os núcleos de nome nunca repetem uma palavra de atributo: o nome
        # "Mouse Óptico" é produzido pelo fragmento do atributo 'optico', de
        # modo que nome, descrição e julgamento de relevância nunca se
        # contradigam.
        base_names=("Mouse", "Mouse Slim", "Mouse Vertical"),
        attributes=(
            "optico", "sem_fio", "bluetooth5", "ergonomico", "gamer",
            "recarregavel", "silencioso", "alta_dpi", "usb_c",
        ),
        specs=(
            "São seis botões programáveis, incluindo avanço e retorno de "
            "página configuráveis pelo aplicativo do fabricante.",
            "A roda de rolagem com passo definido oferece controle preciso em "
            "planilhas longas e documentos extensos.",
            "O corpo pesa apenas noventa gramas, o que reduz o esforço em "
            "movimentos repetidos.",
            "Os pés de PTFE deslizam com suavidade e podem ser substituídos "
            "quando desgastarem.",
        ),
        uses=(
            "É indicado para trabalho em escritório, home office e uso "
            "acadêmico diário.",
            "Atende bem tanto ao usuário doméstico quanto ao profissional que "
            "passa oito horas em frente ao computador.",
            "Funciona com Windows, macOS, Linux e ChromeOS sem necessidade de "
            "driver adicional.",
        ),
        price_range=(29.9, 449.9),
        weight=4.0,
        peripheral=True,
        notebook_ready=True,
    ),
    ProductType(
        key="teclado",
        category="Informática",
        base_names=("Teclado", "Teclado Slim", "Teclado TKL"),
        attributes=(
            "mecanico", "rgb", "sem_fio", "bluetooth5", "gamer", "abnt2",
            "compacto", "hot_swap", "silencioso", "usb_c",
        ),
        specs=(
            "O corpo em liga de alumínio escovado elimina flexão da placa "
            "mesmo sob digitação vigorosa.",
            "O recurso anti-ghosting com N-key rollover registra todas as "
            "teclas pressionadas simultaneamente.",
            "As keycaps em PBT double-shot resistem ao desgaste e não "
            "apagam a serigrafia com o tempo.",
            "Os pés articulados em dois estágios ajustam a inclinação para "
            "cinco ou nove graus.",
        ),
        uses=(
            "Serve para digitação intensa, programação e produção de texto "
            "durante longas jornadas.",
            "É uma escolha comum para quem monta uma estação de trabalho "
            "completa em casa.",
        ),
        price_range=(79.9, 899.9),
        weight=3.5,
        peripheral=True,
        notebook_ready=True,
    ),
    ProductType(
        key="mousepad",
        category="Informática",
        base_names=("Mousepad", "Mousepad Speed", "Mousepad Extenso"),
        attributes=("gamer", "antiderrapante", "impermeavel"),
        specs=(
            "A superfície em microfibra costurada nas bordas evita "
            "desfiamento após lavagens sucessivas.",
            "A espessura de quatro milímetros absorve irregularidades da mesa "
            "e apoia o punho com conforto.",
        ),
        uses=(
            "Acompanha bem qualquer mouse óptico ou a laser e padroniza a "
            "superfície de rastreamento.",
        ),
        price_range=(19.9, 189.9),
        weight=1.0,
        peripheral=True,
        notebook_ready=True,
    ),
    ProductType(
        key="headset",
        category="Informática",
        base_names=("Headset", "Fone de Ouvido", "Headset Over-Ear"),
        attributes=(
            "sem_fio", "bluetooth5", "gamer", "recarregavel", "usb_c",
            "silencioso",
        ),
        specs=(
            "Os drivers de cinquenta milímetros entregam resposta de graves "
            "encorpada sem embolar as frequências médias.",
            "O microfone com cancelamento de ruído isola a voz do barulho "
            "ambiente em chamadas de vídeo.",
            "As almofadas em espuma viscoelástica distribuem a pressão e "
            "evitam dor após horas de uso.",
        ),
        uses=(
            "Atende reuniões remotas, aulas online e sessões prolongadas de "
            "jogo.",
        ),
        price_range=(59.9, 1299.9),
        weight=2.0,
        peripheral=True,
        notebook_ready=True,
    ),
    ProductType(
        key="webcam",
        category="Informática",
        base_names=("Webcam", "Câmera Web", "Webcam Full HD"),
        attributes=("usb_c", "automatico"),
        specs=(
            "A captura em 1080p a sessenta quadros por segundo mantém a "
            "imagem fluida mesmo com movimento.",
            "O foco automático corrige a nitidez em menos de meio segundo "
            "quando o usuário se aproxima.",
            "A tampa de privacidade integrada bloqueia fisicamente a lente "
            "quando a câmera não está em uso.",
        ),
        uses=(
            "Indicada para videoconferência, transmissões ao vivo e gravação "
            "de aulas.",
        ),
        price_range=(89.9, 799.9),
        weight=1.5,
        peripheral=True,
        notebook_ready=True,
    ),
    ProductType(
        key="hub_usb",
        category="Informática",
        base_names=("Hub USB", "Adaptador Multiportas", "Dock Station"),
        attributes=("usb_c", "portatil", "compacto"),
        specs=(
            "São sete portas, incluindo HDMI 4K, leitor de cartão SD e "
            "entrada de rede gigabit.",
            "O chip controlador evita queda de conexão quando vários "
            "dispositivos operam ao mesmo tempo.",
        ),
        uses=(
            "Resolve a escassez de portas dos ultrabooks modernos em uma "
            "única peça.",
        ),
        price_range=(49.9, 599.9),
        weight=1.5,
        peripheral=True,
        notebook_ready=True,
    ),
    ProductType(
        key="notebook",
        category="Informática",
        base_names=("Notebook", "Notebook Ultrafino", "Laptop"),
        attributes=("portatil", "usb_c"),
        specs=(
            "O processador de oito núcleos e os dezesseis gigabytes de "
            "memória dão conta de multitarefa pesada.",
            "O SSD NVMe de um terabyte inicializa o sistema em poucos "
            "segundos.",
            "A tela IPS antirreflexo cobre a maior parte do espectro sRGB, o "
            "que ajuda em trabalho com imagem.",
        ),
        uses=(
            "Serve para estudo, trabalho remoto e edição leve de vídeo.",
        ),
        price_range=(2199.0, 12999.0),
        weight=2.0,
        peripheral=False,
        notebook_ready=False,
    ),
    ProductType(
        key="monitor",
        category="Informática",
        base_names=("Monitor", "Monitor Curvo", "Monitor Ultrawide"),
        attributes=("gamer", "usb_c"),
        specs=(
            "A taxa de atualização de 144 Hz reduz o rastro em movimentos "
            "rápidos na tela.",
            "O painel IPS mantém a fidelidade de cor em ângulos amplos de "
            "visão.",
        ),
        uses=("Complementa a estação de trabalho de quem usa dois monitores.",),
        price_range=(599.0, 4999.0),
        weight=1.5,
        peripheral=True,
        notebook_ready=True,
    ),
    # ----------------------------- Eletrônicos ----------------------------- #
    ProductType(
        key="smartphone",
        category="Eletrônicos",
        base_names=("Smartphone", "Celular", "Smartphone 5G"),
        attributes=("recarregavel", "usb_c", "bluetooth5"),
        specs=(
            "A câmera tripla de cinquenta megapixels conta com estabilização "
            "óptica de imagem.",
            "A bateria de cinco mil miliampères-hora atravessa o dia inteiro "
            "em uso moderado.",
        ),
        uses=("Acompanha a rotina de quem depende do aparelho para trabalhar.",),
        price_range=(799.0, 8999.0),
        weight=2.0,
    ),
    ProductType(
        key="carregador",
        category="Eletrônicos",
        base_names=("Carregador", "Fonte Carregadora", "Carregador Turbo"),
        attributes=("usb_c", "portatil", "automatico"),
        specs=(
            "A entrega de sessenta e cinco watts recarrega notebooks "
            "compactos e celulares pela mesma porta.",
            "A proteção contra sobretensão desliga a saída automaticamente em "
            "caso de pico na rede elétrica.",
        ),
        uses=("Substitui a fonte original sem perda de desempenho.",),
        price_range=(39.9, 399.9),
        weight=1.5,
        notebook_ready=True,
    ),
    ProductType(
        key="bateria_eletronica_pilha",
        category="Eletrônicos",
        base_names=("Bateria de Lítio", "Kit de Pilhas", "Bateria Externa"),
        attributes=("recarregavel", "portatil", "usb_c"),
        specs=(
            "A bateria de íon de lítio mantém oitenta por cento da capacidade "
            "após quinhentos ciclos completos.",
            "O indicador de carga em quatro estágios evita surpresas no meio "
            "do dia.",
        ),
        uses=(
            "Alimenta mouse sem fio, controle remoto, lanterna e outros "
            "aparelhos de baixo consumo.",
        ),
        price_range=(24.9, 349.9),
        weight=1.2,
        extra_terms=("bateria",),
    ),
    ProductType(
        key="tablet",
        category="Eletrônicos",
        base_names=("Tablet", "Tablet Gráfico", "Mesa Digitalizadora"),
        attributes=("bluetooth5", "recarregavel", "usb_c", "portatil"),
        specs=(
            "A área ativa de dez polegadas reconhece oito mil níveis de "
            "pressão da caneta.",
            "A tela de alta densidade exibe texto com nitidez confortável "
            "para leitura prolongada.",
        ),
        uses=("Usado para desenho digital, anotação e consumo de conteúdo.",),
        price_range=(349.0, 5999.0),
        weight=1.5,
        notebook_ready=True,
    ),
    ProductType(
        key="smart_tv",
        category="Eletrônicos",
        base_names=("Smart TV", "Televisor 4K", "Smart TV QLED"),
        attributes=("automatico", "bluetooth5"),
        specs=(
            "O painel 4K com HDR amplia o contraste entre as áreas claras e "
            "escuras da cena.",
            "O sistema operacional embarcado já traz os principais "
            "aplicativos de streaming instalados.",
        ),
        uses=("Centraliza o entretenimento da sala de estar.",),
        price_range=(1499.0, 14999.0),
        weight=1.5,
    ),
    # ------------------------------- Roupas -------------------------------- #
    ProductType(
        key="camiseta",
        category="Roupas",
        base_names=("Camiseta", "Camiseta Básica", "Camiseta Manga Longa"),
        attributes=("impermeavel",),
        specs=(
            "O algodão penteado de fio trinta resulta em toque macio e menor "
            "formação de bolinhas.",
            "A modelagem reta veste bem sem marcar o corpo e mantém o "
            "caimento após várias lavagens.",
            "A manga curta com acabamento em ribana preserva o formato "
            "original da peça.",
        ),
        uses=(
            "Combina com jeans no dia a dia e com alfaiataria leve em "
            "ocasiões informais.",
        ),
        price_range=(29.9, 219.9),
        weight=2.5,
        extra_terms=("manga",),
    ),
    ProductType(
        key="jaqueta",
        category="Roupas",
        base_names=("Jaqueta", "Jaqueta Corta-Vento", "Parka"),
        attributes=("impermeavel", "dobravel", "portatil"),
        specs=(
            "O forro térmico segura o calor corporal sem acrescentar volume "
            "excessivo.",
            "O capuz removível e as mangas com punho elástico bloqueiam a "
            "entrada de vento frio.",
        ),
        uses=("Indicada para o inverno urbano e para viagens a serra.",),
        price_range=(129.9, 899.9),
        weight=1.5,
        extra_terms=("manga",),
    ),
    ProductType(
        key="calca",
        category="Roupas",
        base_names=("Calça Jeans", "Calça Cargo", "Calça de Moletom"),
        attributes=("impermeavel",),
        specs=(
            "O tecido com elastano acompanha o movimento sem deformar o "
            "joelho.",
            "Os cinco bolsos reforçados suportam o peso de carteira e celular "
            "sem rasgar a costura.",
        ),
        uses=("Atende ao uso casual e ao ambiente corporativo mais relaxado.",),
        price_range=(89.9, 549.9),
        weight=2.0,
    ),
    # -------------------------------- Casa --------------------------------- #
    ProductType(
        key="cafeteira",
        category="Casa",
        base_names=("Cafeteira Elétrica", "Cafeteira Expressa", "Cafeteira"),
        attributes=("automatico", "compacto"),
        specs=(
            "A jarra de vidro térmico de um litro e meio serve até doze "
            "xícaras por ciclo.",
            "O sistema corta-pingos permite servir antes do fim da extração "
            "sem sujar a bancada.",
        ),
        uses=("Prepara o café da manhã da família e abastece o escritório.",),
        price_range=(129.9, 2499.0),
        weight=2.0,
    ),
    ProductType(
        key="vela_aromatica",
        category="Casa",
        base_names=("Vela Aromática", "Vela Perfumada", "Kit de Velas"),
        attributes=(),
        specs=(
            "A cera vegetal de soja queima de forma mais limpa e rende até "
            "quarenta horas por unidade.",
            "O pavio de algodão sem chumbo evita fuligem escura no vidro.",
        ),
        uses=("Compõe o ritual de relaxamento no fim do dia.",),
        price_range=(24.9, 249.9),
        weight=1.2,
        extra_terms=("vela",),
    ),
    ProductType(
        key="panela",
        category="Casa",
        base_names=("Jogo de Panelas", "Panela de Pressão", "Frigideira"),
        attributes=("antiderrapante",),
        specs=(
            "O revestimento antiaderente de cerâmica dispensa excesso de óleo "
            "no preparo.",
            "O fundo triplo distribui o calor por igual e evita pontos de "
            "queima no centro.",
        ),
        uses=("Serve tanto para o cozinheiro iniciante quanto para o rotineiro.",),
        price_range=(79.9, 1499.0),
        weight=2.0,
    ),
    ProductType(
        key="luminaria",
        category="Casa",
        base_names=("Luminária de Mesa", "Abajur", "Luminária LED"),
        attributes=("automatico", "usb_c", "rgb"),
        specs=(
            "A temperatura de cor vai de dois mil e setecentos a seis mil e "
            "quinhentos kelvin, do amarelo quente ao branco frio.",
            "O braço articulado direciona a luz exatamente sobre o ponto de "
            "leitura.",
        ),
        uses=("Ilumina a escrivaninha de estudo e a bancada de trabalho.",),
        price_range=(59.9, 699.9),
        weight=1.5,
        notebook_ready=True,
    ),
    # ------------------------------ Esportes ------------------------------- #
    ProductType(
        key="tenis_corrida",
        category="Esportes",
        base_names=("Tênis de Corrida", "Tênis Esportivo", "Tênis de Treino"),
        attributes=("impermeavel", "antiderrapante", "portatil"),
        specs=(
            "A entressola em espuma de alta resiliência devolve energia a "
            "cada passada e reduz o impacto no joelho.",
            "O cabedal em malha respirável mantém o pé seco em treinos "
            "longos sob sol forte.",
            "O amortecimento progressivo protege articulações de quem corre "
            "no asfalto.",
        ),
        uses=("Feito para corrida de rua, esteira e caminhada de longa distância.",),
        price_range=(179.9, 1299.0),
        weight=2.5,
    ),
    ProductType(
        key="bicicleta",
        category="Esportes",
        base_names=("Bicicleta Aro 29", "Mountain Bike", "Bicicleta Urbana"),
        attributes=("dobravel", "antiderrapante"),
        specs=(
            "O quadro em alumínio hidroformado equilibra rigidez e peso "
            "contido.",
            "O câmbio de vinte e uma marchas vence subidas íngremes sem "
            "exigir esforço excessivo.",
        ),
        uses=("Atende ao deslocamento urbano e ao lazer de fim de semana.",),
        price_range=(899.0, 8999.0),
        weight=1.5,
    ),
    ProductType(
        key="barco_vela",
        category="Esportes",
        base_names=("Vela para Barco", "Vela de Optimist", "Kit de Vela Náutica"),
        attributes=("impermeavel",),
        specs=(
            "O tecido em poliéster laminado resiste à radiação solar e à "
            "maresia sem perder o formato.",
            "As costuras duplas reforçadas suportam rajadas fortes durante a "
            "regata.",
        ),
        uses=("Usada em escolas de vela e em competições de classe amadora.",),
        price_range=(499.0, 6999.0),
        weight=0.5,
        extra_terms=("vela",),
    ),
    ProductType(
        key="halteres",
        category="Esportes",
        base_names=("Par de Halteres", "Kit de Pesos", "Halter Ajustável"),
        attributes=("antiderrapante", "compacto"),
        specs=(
            "O revestimento emborrachado protege o piso e reduz o ruído do "
            "apoio.",
            "O sistema de travas rápidas troca a carga em poucos segundos "
            "entre as séries.",
        ),
        uses=("Monta um treino de força completo em casa, sem academia.",),
        price_range=(89.9, 1299.0),
        weight=1.5,
    ),
    # ------------------------------- Livros -------------------------------- #
    ProductType(
        key="livro_tecnico",
        category="Livros",
        base_names=(
            "Livro de Algoritmos", "Guia de Recuperação da Informação",
            "Manual de Estatística Aplicada",
        ),
        attributes=(),
        specs=(
            "A edição revisada traz exercícios resolvidos ao fim de cada "
            "capítulo e código de exemplo comentado.",
            "O volume em capa dura com costura suporta consultas frequentes "
            "sem soltar páginas.",
        ),
        uses=(
            "Adotado em disciplinas de graduação e em programas de pós.",
        ),
        price_range=(49.9, 349.9),
        weight=2.0,
    ),
    ProductType(
        key="livro_ficcao",
        category="Livros",
        base_names=(
            "Romance Contemporâneo", "Coletânea de Contos", "Ficção Científica",
        ),
        attributes=(),
        specs=(
            "A narrativa alterna entre duas linhas temporais que se encontram "
            "no último terço da história.",
            "A tradução direta do original preserva o ritmo seco da prosa "
            "originalmente publicada.",
        ),
        uses=("Boa leitura para o transporte público e para as férias.",),
        price_range=(29.9, 189.9),
        weight=2.0,
    ),
    # ------------------------------ Pet Shop ------------------------------- #
    ProductType(
        key="brinquedo_rato",
        category="Pet Shop",
        base_names=(
            "Brinquedo para Gato Ratinho de Pelúcia",
            "Rato de Pelúcia com Catnip",
            "Kit de Ratinhos de Brinquedo",
        ),
        attributes=(),
        specs=(
            "O ratinho de pelúcia imita o formato e o tamanho de um mouse "
            "real, estimulando o instinto de caça do gato.",
            "O recheio com erva-do-gato mantém o rato atrativo por semanas "
            "após a abertura da embalagem.",
            "A costura reforçada evita que o gato desmonte o rato de pelúcia "
            "nas primeiras brincadeiras.",
        ),
        uses=(
            "Indicado para gatos adultos que passam muitas horas sozinhos em "
            "apartamento.",
        ),
        price_range=(9.9, 89.9),
        weight=1.2,
        extra_terms=("mouse", "rato"),
    ),
    ProductType(
        key="racao",
        category="Pet Shop",
        base_names=("Ração Premium", "Ração Seca para Cães", "Ração para Gatos"),
        attributes=(),
        specs=(
            "A formulação com proteína animal de alta digestibilidade reduz o "
            "volume de fezes.",
            "Os grãos em tamanho reduzido facilitam a mastigação de animais "
            "de porte pequeno.",
        ),
        uses=("Alimentação diária de manutenção para animais adultos.",),
        price_range=(39.9, 499.9),
        weight=1.5,
    ),
    ProductType(
        key="arranhador",
        category="Pet Shop",
        base_names=("Arranhador para Gatos", "Torre Arranhadora", "Poste Arranhador"),
        attributes=("antiderrapante", "dobravel"),
        specs=(
            "O sisal natural enrolado no poste resiste a arranhões diários "
            "sem soltar fiapos.",
            "A base ampla evita tombamento quando o gato sobe com impulso.",
        ),
        uses=("Preserva o sofá e a cortina do desgaste das unhas.",),
        price_range=(69.9, 699.9),
        weight=1.0,
    ),
    # ------------------------ Instrumentos Musicais ------------------------ #
    ProductType(
        key="teclado_musical",
        category="Instrumentos Musicais",
        base_names=(
            "Teclado Musical Arranjador", "Teclado Musical 61 Teclas",
            "Piano Digital",
        ),
        attributes=("bluetooth5", "portatil", "usb_c"),
        specs=(
            "As sessenta e uma teclas sensíveis ao toque reproduzem a "
            "dinâmica do piano acústico.",
            "São mais de seiscentos timbres e duzentos ritmos "
            "acompanhados, do jazz ao forró.",
            "A saída MIDI conecta o instrumento a estações de áudio digital "
            "no computador.",
        ),
        uses=(
            "Atende ao estudante de música e ao tecladista que toca em "
            "igrejas e eventos.",
        ),
        price_range=(599.0, 7999.0),
        weight=1.5,
        extra_terms=("teclado",),
    ),
    ProductType(
        key="bateria_musical",
        category="Instrumentos Musicais",
        base_names=("Bateria Acústica", "Bateria Eletrônica", "Kit de Bateria"),
        attributes=("compacto", "portatil"),
        specs=(
            "A bateria completa traz bumbo, caixa, dois tons, surdo e "
            "pratos de condução e ataque.",
            "Os aros em aço prensado afinam com estabilidade e sustentam a "
            "tensão da pele por mais tempo.",
        ),
        uses=("Serve ao ensaio doméstico e à apresentação em palcos pequenos.",),
        price_range=(1299.0, 12999.0),
        weight=1.0,
        extra_terms=("bateria",),
    ),
    ProductType(
        key="violao",
        category="Instrumentos Musicais",
        base_names=("Violão Clássico", "Violão Eletroacústico", "Guitarra"),
        attributes=("portatil",),
        specs=(
            "O tampo em abeto maciço amadurece o timbre com os anos de uso.",
            "O braço com escala em pau-ferro facilita a execução de acordes "
            "em casas altas.",
        ),
        uses=("Indicado para o iniciante e para o músico de roda de violão.",),
        price_range=(349.0, 6999.0),
        weight=1.2,
    ),
    # ------------------------------ Alimentos ------------------------------ #
    ProductType(
        key="chocolate",
        category="Alimentos",
        base_names=(
            "Tablete de Chocolate Amargo", "Barra de Chocolate ao Leite",
            "Caixa de Bombons",
        ),
        attributes=(),
        specs=(
            "O tablete de setenta por cento cacau tem origem única "
            "rastreada até a fazenda produtora.",
            "A conchagem prolongada resulta em textura sedosa que derrete "
            "uniformemente na boca.",
        ),
        uses=("Ótimo para presente e para acompanhar o café da tarde.",),
        price_range=(9.9, 149.9),
        weight=1.5,
        extra_terms=("tablete",),
    ),
    ProductType(
        key="cafe_graos",
        category="Alimentos",
        base_names=("Café em Grãos", "Café Torrado e Moído", "Cápsulas de Café"),
        attributes=(),
        specs=(
            "Os grãos arábica de torra média preservam a acidez cítrica "
            "característica da região produtora.",
            "A embalagem com válvula unidirecional libera o gás carbônico "
            "sem deixar o oxigênio entrar.",
        ),
        uses=("Rende bem em cafeteira italiana, prensa francesa e coador de pano.",),
        price_range=(19.9, 199.9),
        weight=1.5,
    ),
    ProductType(
        key="manga_fruta",
        category="Alimentos",
        base_names=("Manga Palmer Selecionada", "Caixa de Manga Tommy", "Manga Rosa"),
        attributes=(),
        specs=(
            "A manga é colhida no ponto e amadurece em três dias na "
            "temperatura ambiente.",
            "A polpa firme e pouco fibrosa rende bem em vitaminas e saladas "
            "de fruta.",
        ),
        uses=("Fruta de mesa para o consumo da semana.",),
        price_range=(12.9, 89.9),
        weight=1.0,
        extra_terms=("manga",),
    ),
    # ------------------------------- Beleza -------------------------------- #
    ProductType(
        key="secador",
        category="Beleza",
        base_names=("Secador de Cabelo", "Secador Profissional", "Escova Secadora"),
        attributes=("automatico", "dobravel", "portatil"),
        specs=(
            "O motor de dois mil watts com íons negativos reduz o frizz e "
            "acelera a secagem.",
            "O botão de jato frio fixa o penteado ao fim da escovação.",
        ),
        uses=("Usado em salão e na rotina doméstica de cuidado com os cabelos.",),
        price_range=(89.9, 899.9),
        weight=1.5,
    ),
    ProductType(
        key="perfume",
        category="Beleza",
        base_names=("Perfume Eau de Parfum", "Colônia Masculina", "Perfume Floral"),
        attributes=(),
        specs=(
            "As notas de saída cítricas evoluem para um fundo amadeirado ao "
            "longo de oito horas.",
            "A concentração de vinte por cento de essência garante fixação "
            "prolongada na pele.",
        ),
        uses=("Fragrância versátil para o dia e para a noite.",),
        price_range=(79.9, 899.9),
        weight=1.5,
    ),
    ProductType(
        key="kit_skincare",
        category="Beleza",
        base_names=("Kit de Skincare", "Sérum Facial", "Hidratante Facial"),
        attributes=("automatico",),
        specs=(
            "A fórmula com ácido hialurônico e niacinamida atua na barreira "
            "cutânea sem obstruir os poros.",
            "O produto foi testado dermatologicamente e é livre de parabenos "
            "e fragrância sintética.",
        ),
        uses=("Compõe a rotina de cuidados da manhã e da noite.",),
        price_range=(39.9, 599.9),
        weight=1.5,
    ),
)


# --------------------------------------------------------------------------- #
# Frases que introduzem sinonímia de "periférico" e "notebook"
# --------------------------------------------------------------------------- #

PERIPHERAL_PHRASES: Tuple[str, ...] = (
    "Trata-se de um periférico de entrada projetado para uso contínuo em "
    "estações de trabalho.",
    "Como periférico de computador, dispensa configuração manual e é "
    "reconhecido assim que conectado.",
    "É o tipo de periférico que costuma acompanhar a compra de um "
    "computador novo.",
)

NOTEBOOK_COMPAT_PHRASES: Tuple[str, ...] = (
    "A compatibilidade com notebook é total: basta conectar o receptor na "
    "porta USB do laptop.",
    "Funciona igualmente bem em desktop e em notebook, sem software extra.",
    "Foi pensado para quem trabalha com laptop e precisa de mobilidade entre "
    "escritório e casa.",
    "O acessório acompanha quem leva o notebook para cafés e espaços de "
    "coworking.",
)


# --------------------------------------------------------------------------- #
# Geração
# --------------------------------------------------------------------------- #


@dataclass
class GeneratedProduct:
    """
    Produto gerado, com os atributos internos usados nos julgamentos.

    Attributes:
        record: Dicionário exatamente no schema exigido pelo corpus.
        ptype: Chave canônica do tipo de produto.
        attrs: Conjunto de atributos sorteados para o item.
        peripheral: Se é um periférico de computador.
        notebook_ready: Se é indicado para uso com notebook.
    """

    record: Dict
    ptype: str
    attrs: Set[str]
    peripheral: bool
    notebook_ready: bool


def _fold(word: str) -> str:
    """
    Normaliza uma palavra para comparação (caixa baixa, sem acento).

    Args:
        word: Palavra de entrada.

    Returns:
        Forma normalizada, usada para detectar repetições no nome.

    Exemplo:
        >>> _fold("Óptico")
        'optico'
    """
    decomposed = unicodedata.normalize("NFKD", word.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _weighted_type_for(index: int, rng: random.Random) -> ProductType:
    """
    Sorteia o tipo de produto respeitando os pesos declarados.

    Args:
        index: Posição do produto no catálogo (não usada no sorteio, mantida
            para clareza da assinatura).
        rng: Gerador aleatório já semeado para este produto.

    Returns:
        O :class:`ProductType` escolhido.
    """
    weights = [ptype.weight for ptype in PRODUCT_TYPES]
    return rng.choices(PRODUCT_TYPES, weights=weights, k=1)[0]


def _build_name(ptype: ProductType, attrs: Sequence[str], rng: random.Random) -> str:
    """
    Compõe o nome comercial do produto.

    Args:
        ptype: Tipo de produto.
        attrs: Atributos sorteados.
        rng: Gerador aleatório do produto.

    Returns:
        Nome no formato ``"<núcleo> <atributos> <marca> <cor?>"``.

    Exemplo:
        >>> rng = random.Random(0)
        >>> name = _build_name(PRODUCT_TYPES[0], ["sem_fio"], rng)
        >>> "Sem Fio" in name
        True
    """
    core = rng.choice(ptype.base_names)
    core_words = {_fold(word) for word in core.split()}
    # "Teclado Gamer" + fragmento "Gamer" produziria "Teclado Gamer Gamer";
    # descarta-se o que o núcleo do nome já diz.
    fragments = [
        ATTRIBUTE_NAME_FRAGMENT[attr]
        for attr in attrs
        if attr in ATTRIBUTE_NAME_FRAGMENT
        and _fold(ATTRIBUTE_NAME_FRAGMENT[attr].split()[0]) not in core_words
    ]
    rng.shuffle(fragments)
    fragments = fragments[: rng.randint(0, min(2, len(fragments)))]

    parts: List[str] = [core, *fragments, rng.choice(BRANDS)]
    if rng.random() < 0.35:
        parts.append(rng.choice(COLORS).capitalize())
    return " ".join(parts)


def _sentence_pool(
    ptype: ProductType,
    attrs: Sequence[str],
    name: str,
    rng: random.Random,
) -> List[str]:
    """
    Monta o repertório de frases disponíveis para a descrição de um produto.

    A ordem é intencional: abertura, atributos, especificações, usos,
    sinonímia e, por último, frases genéricas de preenchimento.

    Args:
        ptype: Tipo de produto.
        attrs: Atributos sorteados.
        name: Nome comercial já composto.
        rng: Gerador aleatório do produto.

    Returns:
        Lista ordenada de frases candidatas.
    """
    adjective = rng.choice(QUALITY_ADJECTIVES)
    opening = (
        f"O {name} é um produto {adjective} da categoria {ptype.category}, "
        f"desenvolvido para entregar desempenho consistente no uso cotidiano."
    )

    sentences: List[str] = [opening]

    for attr in attrs:
        phrases = ATTRIBUTE_PHRASES.get(attr)
        if phrases:
            sentences.append(rng.choice(phrases))

    specs = list(ptype.specs)
    rng.shuffle(specs)
    sentences.extend(specs)

    uses = list(ptype.uses)
    rng.shuffle(uses)
    sentences.extend(uses)

    # Sinonímia: só uma fração dos periféricos usa a palavra "periférico",
    # de modo que a Q5 ("periférico para notebook") seja um teste real de
    # correspondência semântica e não de casamento literal.
    if ptype.peripheral and rng.random() < 0.35:
        sentences.append(rng.choice(PERIPHERAL_PHRASES))
    if ptype.notebook_ready and rng.random() < 0.60:
        sentences.append(rng.choice(NOTEBOOK_COMPAT_PHRASES))

    fillers = list(FILLER_SENTENCES)
    rng.shuffle(fillers)
    sentences.extend(fillers)

    closers = list(GENERIC_CLOSERS)
    rng.shuffle(closers)
    sentences.extend(closers)

    return sentences


def _build_description(
    ptype: ProductType,
    attrs: Sequence[str],
    name: str,
    rng: random.Random,
) -> str:
    """
    Gera a descrição do produto respeitando a faixa de comprimento alvo.

    Args:
        ptype: Tipo de produto.
        attrs: Atributos sorteados.
        name: Nome comercial.
        rng: Gerador aleatório do produto.

    Returns:
        Descrição com comprimento entre ``config.DESCRIPTION_MIN_WORDS`` e
        aproximadamente ``config.DESCRIPTION_MAX_WORDS`` palavras.
    """
    target = rng.randint(config.DESCRIPTION_MIN_WORDS, config.DESCRIPTION_MAX_WORDS)
    pool = _sentence_pool(ptype, attrs, name, rng)

    chosen: List[str] = []
    word_count = 0
    cursor = 0
    while word_count < target:
        if cursor >= len(pool):
            # Reembaralha o preenchimento genérico para alcançar o alvo sem
            # repetir a mesma sequência de frases.
            extra = list(FILLER_SENTENCES) + list(GENERIC_CLOSERS)
            rng.shuffle(extra)
            pool.extend(extra)
        sentence = pool[cursor]
        cursor += 1
        chosen.append(sentence)
        word_count += len(sentence.split())

    # A última frase pode estourar o teto; remove-se enquanto isso não violar
    # o piso de comprimento, mantendo a descrição na faixa contratada.
    while len(chosen) > 1 and word_count > config.DESCRIPTION_MAX_WORDS:
        candidate = word_count - len(chosen[-1].split())
        if candidate < config.DESCRIPTION_MIN_WORDS:
            break
        word_count = candidate
        chosen.pop()

    return " ".join(chosen)


def _make_product(product_id: int, seed: int) -> GeneratedProduct:
    """
    Gera um único produto de forma determinística.

    Cada produto recebe seu próprio ``random.Random`` semeado a partir do
    ``product_id``. Isso torna o catálogo um *superconjunto estável*:
    ``generate_corpus(10000)[:5000] == generate_corpus(5000)``.

    Args:
        product_id: Identificador sequencial, começando em 1.
        seed: Semente global do experimento.

    Returns:
        O produto gerado e seus metadados internos.
    """
    rng = random.Random(seed * 1_000_003 + product_id)

    ptype = _weighted_type_for(product_id, rng)

    available = list(ptype.attributes)
    rng.shuffle(available)
    n_attrs = rng.randint(0, min(4, len(available)))
    attrs = sorted(available[:n_attrs])

    # Polissemia deliberada: uma minoria dos teclados musicais menciona
    # iluminação RGB, criando falsos positivos plausíveis para a Q4.
    if ptype.key == "teclado_musical" and rng.random() < 0.12:
        attrs = sorted(set(attrs) | {"rgb"})

    name = _build_name(ptype, attrs, rng)
    description = _build_description(ptype, attrs, name, rng)

    low, high = ptype.price_range
    price = round(rng.uniform(low, high), 2)
    rating = round(min(5.0, max(1.0, rng.gauss(4.1, 0.6))), 1)

    record = {
        "product_id": product_id,
        "name": name,
        "description": description,
        "category": ptype.category,
        "price": price,
        "rating": rating,
    }

    return GeneratedProduct(
        record=record,
        ptype=ptype.key,
        attrs=set(attrs),
        peripheral=ptype.peripheral,
        notebook_ready=ptype.notebook_ready,
    )


def generate_products(
    size: int = config.CORPUS_SIZE,
    seed: int = config.RANDOM_SEED,
) -> List[GeneratedProduct]:
    """
    Gera ``size`` produtos com seus metadados internos.

    Args:
        size: Quantidade de produtos.
        seed: Semente global (reprodutibilidade).

    Returns:
        Lista de :class:`GeneratedProduct` com ``product_id`` de 1 a ``size``.

    Exemplo:
        >>> items = generate_products(3)
        >>> [item.record["product_id"] for item in items]
        [1, 2, 3]
    """
    logger.info("Gerando %d produtos (seed=%d)...", size, seed)
    return [_make_product(pid, seed) for pid in range(1, size + 1)]


def generate_corpus(
    size: int = config.CORPUS_SIZE,
    seed: int = config.RANDOM_SEED,
) -> List[Dict]:
    """
    Gera o corpus no schema final, sem os metadados internos.

    Args:
        size: Quantidade de produtos.
        seed: Semente global.

    Returns:
        Lista de dicionários com as chaves ``product_id``, ``name``,
        ``description``, ``category``, ``price`` e ``rating``.

    Exemplo:
        >>> corpus = generate_corpus(2)
        >>> sorted(corpus[0])
        ['category', 'description', 'name', 'price', 'product_id', 'rating']
    """
    return [item.record for item in generate_products(size, seed)]


# --------------------------------------------------------------------------- #
# Julgamentos de relevância (qrels)
# --------------------------------------------------------------------------- #

PERIPHERAL_TYPES: frozenset = frozenset(
    {"mouse", "teclado", "mousepad", "headset", "webcam", "hub_usb", "monitor"}
)


def _judge_q1(item: GeneratedProduct) -> int:
    """Julga ``mouse``: periférico é totalmente relevante; mousepad, parcial."""
    if item.ptype == "mouse":
        return 3
    if item.ptype == "mousepad":
        return 1
    return 0


def _judge_q2(item: GeneratedProduct) -> int:
    """Julga ``mouse óptico sem fio`` por cobertura de atributos."""
    if item.ptype != "mouse":
        return 0
    wireless = {"sem_fio", "bluetooth5"} & item.attrs
    if "optico" in item.attrs and wireless:
        return 3
    if "optico" in item.attrs or wireless:
        return 2
    return 1


def _judge_q3(item: GeneratedProduct) -> int:
    """Julga ``mouse ergonômico bluetooth 5.0`` por cobertura de atributos."""
    if item.ptype != "mouse":
        return 0
    hits = len({"ergonomico", "bluetooth5"} & item.attrs)
    if hits == 2:
        return 3
    if hits == 1:
        return 2
    return 1


def _judge_q4(item: GeneratedProduct) -> int:
    """Julga ``teclado AND (mecânico OR rgb) AND NOT gamer``."""
    if item.ptype != "teclado":
        return 0
    if "gamer" in item.attrs:
        return 0
    if {"mecanico", "rgb"} & item.attrs:
        return 3
    return 0


def _judge_q5(item: GeneratedProduct) -> int:
    """Julga ``periférico para notebook``: periférico compatível vale mais."""
    if item.ptype not in PERIPHERAL_TYPES:
        return 0
    return 3 if item.notebook_ready else 1


JUDGES = {
    "Q1": _judge_q1,
    "Q2": _judge_q2,
    "Q3": _judge_q3,
    "Q4": _judge_q4,
    "Q5": _judge_q5,
}


def build_qrels(items: Sequence[GeneratedProduct]) -> Dict[str, Dict[str, int]]:
    """
    Deriva os julgamentos de relevância graduados a partir dos metadados.

    Como o corpus é sintético, a relevância é conhecida por construção: não
    há necessidade de anotação manual e a avaliação fica objetiva.

    Args:
        items: Produtos gerados, com metadados internos.

    Returns:
        Mapa ``{query_id: {product_id (str): grau 1-3}}``. Documentos com grau
        zero são omitidos.

    Exemplo:
        >>> qrels = build_qrels(generate_products(50))
        >>> set(qrels) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
        True
    """
    qrels: Dict[str, Dict[str, int]] = {qid: {} for qid in JUDGES}
    for item in items:
        for qid, judge in JUDGES.items():
            grade = judge(item)
            if grade > 0:
                qrels[qid][str(item.record["product_id"])] = grade
    for qid, judgments in qrels.items():
        logger.info("qrels %s: %d documentos relevantes.", qid, len(judgments))
    return qrels


# --------------------------------------------------------------------------- #
# Entrada / saída
# --------------------------------------------------------------------------- #


def save_corpus(corpus: Sequence[Dict], path: Path = config.PRODUCTS_PATH) -> None:
    """
    Grava o corpus em JSON UTF-8 indentado.

    Args:
        corpus: Lista de produtos.
        path: Caminho de destino.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(corpus), handle, ensure_ascii=False, indent=2)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(
        "Corpus salvo em %s (%d produtos, %.1f MB).", path, len(corpus), size_mb
    )


def save_qrels(
    qrels: Dict[str, Dict[str, int]], path: Path = config.QRELS_PATH
) -> None:
    """
    Grava os julgamentos de relevância em JSON.

    Args:
        qrels: Mapa de julgamentos.
        path: Caminho de destino.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(qrels, handle, ensure_ascii=False, indent=2)
    logger.info("Julgamentos de relevância salvos em %s.", path)


def load_corpus(path: Path = config.PRODUCTS_PATH) -> List[Dict]:
    """
    Carrega o corpus do disco, gerando-o caso ainda não exista.

    Args:
        path: Caminho do arquivo JSON.

    Returns:
        Lista de produtos.

    Raises:
        ValueError: Se o arquivo existir mas não contiver uma lista.
    """
    if not path.exists():
        logger.warning("%s não encontrado; gerando corpus agora.", path)
        corpus = generate_corpus()
        save_corpus(corpus, path)
        return corpus

    with path.open("r", encoding="utf-8") as handle:
        corpus = json.load(handle)
    if not isinstance(corpus, list):
        raise ValueError(f"Conteúdo inesperado em {path}: esperava uma lista.")
    logger.info("Corpus carregado de %s (%d produtos).", path, len(corpus))
    return corpus


def load_qrels(path: Path = config.QRELS_PATH) -> Dict[str, Dict[str, int]]:
    """
    Carrega os julgamentos de relevância do disco.

    Args:
        path: Caminho do arquivo JSON.

    Returns:
        Mapa ``{query_id: {product_id: grau}}``; vazio se o arquivo não existir.
    """
    if not path.exists():
        logger.warning("%s não encontrado; avaliação quantitativa ficará vazia.", path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def corpus_statistics(corpus: Sequence[Dict]) -> Dict[str, object]:
    """
    Calcula estatísticas descritivas do corpus.

    Args:
        corpus: Lista de produtos.

    Returns:
        Dicionário com contagem, distribuição por categoria e estatísticas de
        comprimento das descrições.
    """
    lengths = [len(product["description"].split()) for product in corpus]
    categories: Dict[str, int] = {}
    for product in corpus:
        categories[product["category"]] = categories.get(product["category"], 0) + 1

    return {
        "n_produtos": len(corpus),
        "palavras_min": min(lengths) if lengths else 0,
        "palavras_max": max(lengths) if lengths else 0,
        "palavras_media": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "total_palavras": sum(lengths),
        "categorias": dict(sorted(categories.items(), key=lambda kv: -kv[1])),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """
    Ponto de entrada da geração do corpus.

    Args:
        argv: Argumentos de linha de comando (usa ``sys.argv`` se ``None``).

    Returns:
        Código de saída do processo.
    """
    parser = argparse.ArgumentParser(
        description="Gera o corpus sintético de e-commerce."
    )
    parser.add_argument(
        "--size", type=int, default=config.CORPUS_SIZE, help="Número de produtos."
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED, help="Semente aleatória."
    )
    parser.add_argument(
        "--output", type=Path, default=config.PRODUCTS_PATH, help="Arquivo de saída."
    )
    args = parser.parse_args(argv)

    config.setup_logging()
    config.ensure_directories()

    items = generate_products(args.size, args.seed)
    corpus = [item.record for item in items]
    save_corpus(corpus, args.output)
    save_qrels(build_qrels(items))

    stats = corpus_statistics(corpus)
    logger.info(
        "Descrições: %d-%d palavras (média %.1f); vocabulário bruto de %d palavras.",
        stats["palavras_min"],
        stats["palavras_max"],
        stats["palavras_media"],
        stats["total_palavras"],
    )
    for category, count in stats["categorias"].items():
        logger.info("  %-24s %5d produtos", category, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
