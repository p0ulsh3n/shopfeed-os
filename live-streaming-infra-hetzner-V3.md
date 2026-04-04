# Live Streaming Infrastructure — Hetzner + OVH + CDN
## Architecture de Production Optimisée : Phase 1 → Phase 3

> **Référentiel :** Senior Infra Engineer · Pattern TikTok / Taobao Live
> **Stack :** HaishinKit → RTMP/SRT → HAProxy → SRS Origin (OVH Rise-1) → FFmpeg **Intel QSV** (Hetzner EX44) → SRS Edge (OVH Rise-2 2Gbps) → Gcore CDN + Bunny CDN → Viewers
> **Version :** 4.0 · Avril 2026 · 4 profils ABR TikTok-exact (`_ld5` / `_sd5` / `_zsd5` / `_hd5`)

---

## Technologies & Références

| Composant | Technologie | Lien officiel |
|-----------|------------|---------------|
| **Media Server** | SRS (Simple Realtime Server) v6.x | https://ossrs.net · https://github.com/ossrs/srs |
| **Ingest Mobile iOS** | HaishinKit 2.2.5 (RTMP + SRT) | https://github.com/HaishinKit/HaishinKit.swift |
| **Ingest Mobile Android** | HaishinKit.kt (RTMP) | https://github.com/HaishinKit/HaishinKit.kt |
| **Transcodage GPU** | FFmpeg + Intel QSV (h264_qsv) | https://ffmpeg.org/ffmpeg-codecs.html#QSV-Encoders |
| **Load Balancer** | HAProxy | https://www.haproxy.org |
| **CDN EU/USA** | Bunny CDN | https://bunny.net |
| **CDN Afrique/Asie** | Gcore CDN | https://gcore.com/cdn |
| **Serveurs transcodage** | Hetzner EX44 (i5-13500 QSV) | https://www.hetzner.com/dedicated-rootserver/ex44 |
| **Serveur Origin** | OVH Rise-1 | https://eco.ovhcloud.com/fr/hosted-private-cloud/ |
| **Serveurs Edge** | OVH Rise-2 + 2 Gbps | https://eco.ovhcloud.com/fr/hosted-private-cloud/ |
| **Geo DNS** | Cloudflare Load Balancer | https://developers.cloudflare.com/load-balancing |

> **SRS (Simple Realtime Server)** : Serveur RTMP/HTTP-FLV/WebRTC open-source, équivalent cloud-grade développé par une équipe chinoise (ex-Alibaba). C'est le même stack utilisé par Taobao Live. Version 6.x stable recommandée — doc complète sur https://ossrs.net/lts/en-us/

---

## Intel QSV — Le choix technique EX44

L'Intel i5-13500 possède l'**UHD Graphics 770**, avec **2 Multi-Format Codec Engines (MFX) indépendants**. Ce n'est pas une "iGPU" grand-public, c'est un ASIC (Application-Specific Integrated Circuit) dédié au transcodage vidéo.

> **Capacité hardware confirmée 2026 :** L'UHD 770 peut encoder jusqu'à 15-20 flux **720p** simultanément en production — le hardware est théoriquement capable de 1080p mais **nous ne l'utilisons pas en production** (voir note ci-dessous). Tests montrent des vitesses **8x-10x** la vitesse réelle pour un flux single.

> Attention : **Note 1080p :** Le 1080p n'est PAS dans nos profils de production. Nos 4 profils (`_ld5` 360p / `_sd5` 480p / `_zsd5` 720p / `_hd5` 720p) couvrent 100% des usages mobiles. Le 1080p sera envisagé uniquement si un marché mâture le justifie — il demanderait ~5-6 Mbps/stream et réduirait la capacité par EX44 à ~10 streams actifs.

- **2 MFX Engines indépendants :** Chaque moteur gère sa propre file de streams. Les 2 ensemble sur l'UHD 770 permettent le pipeline multi-stream sans contention.
- **Performance réelle :** 1 serveur EX44 = 15 à 20 flux complets `_ld5`+`_sd5`+`_zsd5`+`_hd5` simultanés, CPU à <10%
- **Low Latency absolu :** `-async_depth 1` + `-look_ahead 0` réduisent le délai RTMP→FLV au minimum
- **Stabilité thermique :** Le moteur QSV intégré ne "throttle" jamais sous charge 24h/24 continue
- **Cout :** 49.00 EUR/mois (47.30 + 1.70 IPv4) + 99 EUR setup -- soit ~3.3 EUR/stream/mois (15 streams actifs)
- **AV1 : décodage OK, encodage NON** — l'i5-13500 décode l'AV1 hardware mais ne peut pas l'encoder. On reste sur H.264 QSV pour l'output live.

### 4. Drivers Linux requis (Red Flag à éviter)

```bash
# Ubuntu 22.04/24.04 — OBLIGATOIRE sur chaque EX44 avant de lancer FFmpeg
# Sans ces 2 packages, FFmpeg ne détecte pas QSV et tombera en soft CPU = catastrophe
apt-get update && apt-get install -y \
    intel-media-va-driver-non-free \
    libmfx1 \
    vainfo \
    libva-drm2

# Ajouter l'utilisateur ffmpeg au groupe render (accès /dev/dri sans root)
usermod -aG render $(whoami)

# Vérifier que les 2 MFX Engines sont détectés :
vainfo | grep H264
# → Doit retourner : VAProfileH264Main, VAProfileH264High, VAProfileH264ConstrainedBaseline

# Vérifier que h264_qsv est disponible dans votre build FFmpeg :
ffmpeg -encoders 2>/dev/null | grep qsv
# → Doit lister : h264_qsv, hevc_qsv, vp9_qsv...

# Vérifier que vpp_qsv (filtre critique pour split multi-output) est dispo :
ffmpeg -filters 2>/dev/null | grep qsv
# → Doit lister : vpp_qsv, scale_qsv

# Noyau Linux 6.x minimum (EX44 livré avec Ubuntu 24.04 par défaut chez Hetzner)
uname -r  # Doit retourner 6.x
```

### 5. Commande FFmpeg QSV — Production Grade (Intuable)

```bash
#!/bin/bash
# /opt/livestream/transcode-qsv.sh
# Args: $1=STREAM_KEY $2=SRS_ORIGIN_IP $3=SRS_EDGE_IP

set -euo pipefail
STREAM_KEY="${1}"
SRS_ORIGIN="${2}"
SRS_EDGE="${3}"

# Attendre que le flux source soit disponible
RETRY=0
while ! ffprobe -v quiet -i "rtmp://${SRS_ORIGIN}/live/${STREAM_KEY}" 2>/dev/null; do
    RETRY=$((RETRY + 1)); [ $RETRY -ge 5 ] && { echo "[ERROR] Source unreachable: ${STREAM_KEY}"; exit 1; }
    sleep 1
done

exec ffmpeg \
  -loglevel warning \
  -hwaccel qsv \
  -hwaccel_output_format qsv \
  -qsv_device /dev/dri/renderD128 \
  -c:v h264_qsv \
  -i "rtmp://${SRS_ORIGIN}/live/${STREAM_KEY}" \
  \
  -filter_complex \
  "[0:v]split=4[vld5][vsd5][vzsd5][vhd5]; \
   [vhd5]vpp_qsv=w=1280:h=720[out_hd5]; \
   [vzsd5]vpp_qsv=w=1280:h=720[out_zsd5]; \
   [vsd5]vpp_qsv=w=854:h=480[out_sd5]; \
   [vld5]vpp_qsv=w=640:h=360[out_ld5]" \
  \
  -map "[out_hd5]" \
  -c:v h264_qsv -preset veryfast -async_depth 1 -look_ahead 0 \
  -profile:v high -level 4.0 \
  -b:v 3000k -maxrate 3500k -bufsize 7000k \
  -r 30 -g 30 -sc_threshold 0 \
  -map 0:a -c:a aac -b:a 128k -ar 44100 -ac 2 \
  -f flv "rtmp://${SRS_EDGE}/live/${STREAM_KEY}_hd5" \
  \
  -map "[out_zsd5]" \
  -c:v h264_qsv -preset veryfast -async_depth 1 -look_ahead 0 \
  -profile:v main -level 3.2 \
  -b:v 1800k -maxrate 2100k -bufsize 4200k \
  -r 30 -g 30 -sc_threshold 0 \
  -map 0:a -c:a aac -b:a 128k -ar 44100 -ac 2 \
  -f flv "rtmp://${SRS_EDGE}/live/${STREAM_KEY}_zsd5" \
  \
  -map "[out_sd5]" \
  -c:v h264_qsv -preset veryfast -async_depth 1 -look_ahead 0 \
  -profile:v main -level 3.1 \
  -b:v 1000k -maxrate 1200k -bufsize 2400k \
  -r 30 -g 30 -sc_threshold 0 \
  -map 0:a -c:a aac -b:a 96k -ar 44100 -ac 2 \
  -f flv "rtmp://${SRS_EDGE}/live/${STREAM_KEY}_sd5" \
  \
  -map "[out_ld5]" \
  -c:v h264_qsv -preset veryfast -async_depth 1 -look_ahead 0 \
  -profile:v baseline -level 3.0 \
  -b:v 500k -maxrate 600k -bufsize 1200k \
  -r 30 -g 30 -sc_threshold 0 \
  -map 0:a -c:a aac -b:a 64k -ar 44100 -ac 2 \
  -f flv "rtmp://${SRS_EDGE}/live/${STREAM_KEY}_ld5"
```

**Notes critiques QSV (Recherche FFmpeg docs 2025 + feed_engine_architecture_v4 confirmés) :**
- **4 profils TikTok-exact** : `_ld5` / `_sd5` / `_zsd5` / `_hd5` (naming reverse-engineered depuis les URLs CDN de TikTok/Douyin)
- `profile:v baseline/main/high` : niveaux H.264 corrects selon la qualité (compatibilité appareils bas de gamme pour `_ld5`)
- `vpp_qsv` dans `filter_complex` : CORRECT — `scale_qsv` ne fonctionne pas avec `split=4` multi-output
- `-hwaccel_output_format qsv` : frames décodés restent en mémoire GPU (aucune copie vers RAM)
- `-look_ahead 0` : CRITIQUE — élimine 500ms-2s de latence buffer
- `-g 30` (1 keyframe/s à 30fps) : standard live — le viewer rejoint le stream à moins d'1s d'attente
- `-async_depth 1` : latence pipeline minimale → live interactif fluide

---

## Schémas Architecture

> **Pour draw.io :** Copier le texte des schémas ci-dessous et reconstituer les boîtes/flèches. Les specs exactes sont indiquées dans chaque nœud.

---

### Schéma 1 — Flux de données complet (Data Flow)

```
┌───────┐
│                         STREAMERS (Mobiles)                             │
│                                                                         │
│   ┌──────────────────────┐         ┌──────────────────────┐            │
│   │   iOS (HaishinKit    │         │  Android (HaishinKit │            │
│   │   2.2.5 swift)       │         │  .kt)                │            │
│   │   SRT (priorité)     │         │  RTMP uniquement     │            │
│   │   RTMP (fallback)    │         │                      │            │
│   └──────────┬───────────┘         └──────────┬───────────┘            │
└──────────────┼──────────────────────────────── ┼───────────────────────┘
               │ SRT :10080                       │ RTMP :1935
               └────────────────┬─────────────────┘
                                ▼
               ┌────────────────────────────────┐
               │    HAProxy — Load Balancer      │
               │    OVH Rise-1 · :1935 / :10080  │
               │    balance source (consistent   │
               │    hash par IP streamer)        │
               └────────────────┬───────────────┘
                                │ RTMP :11935
                                ▼
               ┌────────────────────────────────┐
               │     SRS Origin (ossrs.net)      │
               │     OVH Rise-1                  │
               │     Xeon-E 2336 · 32 GB         │
               │     listen :11935               │
               │     ├─ on_publish → Orchestrateur│
               │     └─ forward → Orchestrateur  │
               └────────────────┬───────────────┘
                                │ RTMP (stream brut)
                                ▼
               ┌────────────────────────────────┐
               │  Orchestrateur Flask + Redis    │
               │  OVH Rise-1 · port :8085        │
               │  Redis CPX11 · 10.0.4.1         │
               │  → sélectionne EX44 disponible  │
               │  → spawn FFmpeg sur le worker   │
               └────────────────┬───────────────┘
                                │ Dispatch RTMP → worker
                                ▼
     ┌──────────────────────────────────────────────────────┐
     │              FFmpeg QSV Cluster                       │
     │              20× Hetzner EX44                        │
     │              Intel i5-13500 · QSV UHD 770 · 64 GB   │
     │                                                       │
     │  ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐    │
     │  │ EX44-1 │ │ EX44-2 │ │ EX44-3 │ ... │EX44-20 │    │
     │  │15 str  │ │15 str  │ │15 str  │     │15 str  │    │
     │  └────────┘ └────────┘ └────────┘     └────────┘    │
     │                                                       │
     │  Chaque EX44 : split=4 → vpp_qsv → h264_qsv          │
     │   _ld5  360p  500 kbps  baseline                     │
     │   _sd5  480p 1000 kbps  main                         │
     │   _zsd5 720p 1800 kbps  main                         │
     │   _hd5  720p 3000 kbps  high                         │
     └──────────────────────────┬───────────────────────────┘
                                │ RTMP 4× qualités
                                │ (via vRack Hetzner → OVH)
                                ▼
     ┌──────────────────────────────────────────────────────┐
     │           SRS Edge Cluster (ossrs.net)                │
     │           3× OVH Rise-2                              │
     │           Xeon-E 2388G · 64 GB · 2 Gbps garanti      │
     │                                                       │
     │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
     │  │   Edge #1    │ │   Edge #2    │ │   Edge #3    │ │
     │  │ 2 Gbps out   │ │ 2 Gbps out   │ │ 2 Gbps out   │ │
     │  │ HTTP-FLV :80 │ │ HTTP-FLV :80 │ │ HTTP-FLV :80 │ │
     │  └──────────────┘ └──────────────┘ └──────────────┘ │
     │                                                       │
     │  Firewall WHITELIST : seuls Gcore + Bunny IPs     │
     │     autorisés en entrée — DDoS = 0 impact            │
     └──────────────────────────┬───────────────────────────┘
                                │ HTTP-FLV origin pull
                                ▼
               ┌────────────────────────────────┐
               │  Cloudflare Geo DNS / LB        │
               │  Geo-IP steering < 20ms         │
               └────────┬───────────────┬───────┘
                        │               │
          ┌─────────────▼──┐      ┌─────▼──────────────┐
          │ Afrique + Asie │      │   Europe + USA      │
          └─────────────┬──┘      └─────┬───────────────┘
                        ▼               ▼
          ┌─────────────────┐  ┌─────────────────────┐
          │   Gcore CDN     │  │     Bunny CDN        │
          │  210+ PoPs      │  │    119 PoPs          │
          │  200 Tbps       │  │    250 Tbps          │
          │  €0.020/GB      │  │    $0.005/GB (vol)   │
          └────────┬────────┘  └────────┬────────────┘
                   └────────┬───────────┘
                            ▼
          ┌──────────────────────────────────────┐
          │            VIEWERS                    │
          │  libVLC (Android) / MobileVLCKit (iOS)│
          │  HTTP-FLV · ABR dual-player           │
          │  selectBestQuality() → _ld5/_sd5/     │
          │  _zsd5/_hd5 selon NetworkMonitor      │
          │  abr_pts = switch seamless (0 freeze) │
          └──────────────────────────────────────┘
```

---

### Schéma 2 — Infrastructure physique & réseau

```
╔══════════════════════════════════════════════════════════════════════╗
║                     DATACENTER OVH (Europe)                          ║
║                                                                       ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │                    OVH Rise-1                                │     ║
║  │   Xeon-E 2336 · 32 GB RAM · 1 Gbps port                     │     ║
║  │   IP publique : <ORIGIN_PUBLIC_IP>                           │     ║
║  │   IP vRack   : 10.0.1.1                                      │     ║
║  │                                                               │     ║
║  │   ╔═══════════════╗  ╔══════════════════╗  ╔══════════════╗  │     ║
║  │   ║    HAProxy    ║  ║   SRS Origin     ║  ║ Orchestrateur║  │     ║
║  │   ║ :1935 / :10080║  ║  (ossrs.net v6)  ║  ║ Flask :8085  ║  │     ║
║  │   ║ balance source║  ║    :11935 RTMP   ║  ║ Redis :6379  ║  │     ║
║  │   ╚═══════════════╝  ╚══════════════════╝  ╚══════════════╝  │     ║
║  └─────────────────────────────────────────────────────────────┘     ║
║                               │ vRack OVH (privé, gratuit)           ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │          3× OVH Rise-2  (SRS Edge Cluster)                   │    ║
║  │                                                               │    ║
║  │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐ │    ║
║  │  │    Rise-2 #1     │ │    Rise-2 #2     │ │   Rise-2 #3  │ │    ║
║  │  │ Xeon-E 2388G     │ │ Xeon-E 2388G     │ │ Xeon-E 2388G │ │    ║
║  │  │ 64 GB · 2 Gbps   │ │ 64 GB · 2 Gbps   │ │ 64 GB · 2Gbps│ │    ║
║  │  │ IP vRack:10.0.2.1│ │ IP vRack:10.0.2.2│ │ IP:10.0.2.3  │ │    ║
║  │  │ SRS Edge :1935   │ │ SRS Edge :1935   │ │ SRS Edge     │ │    ║
║  │  │ HTTP-FLV  :8080  │ │ HTTP-FLV  :8080  │ │ HTTP-FLV     │ │    ║
║  │  │                  │ │                  │ │              │ │    ║
║  │  │ FIREWALL      │ │ FIREWALL      │ │ FIREWALL  │ │    ║
║  │  │ INPUT DROP tout  │ │ INPUT DROP tout  │ │ INPUT DROP   │ │    ║
║  │  │ ALLOW Gcore IPs  │ │ ALLOW Gcore IPs  │ │ ALLOW Gcore  │ │    ║
║  │  │ ALLOW Bunny IPs  │ │ ALLOW Bunny IPs  │ │ ALLOW Bunny  │ │    ║
║  │  │ ALLOW 10.0.1.x   │ │ ALLOW 10.0.1.x   │ │ ALLOW 10.0.1 │ │    ║
║  │  └──────────────────┘ └──────────────────┘ └──────────────┘ │    ║
║  └──────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════╝
                    │ Internet (traffic FFmpeg → Edge)
╔══════════════════════════════════════════════════════════════════════╗
║                   DATACENTER HETZNER (Europe)                        ║
║                                                                       ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │              20× Hetzner EX44 — FFmpeg Cluster              │     ║
║  │                                                               │     ║
║  │  ┌──────────┐ ┌──────────┐ ┌──────────┐   ┌──────────┐     │     ║
║  │  │ EX44 #1  │ │ EX44 #2  │ │ EX44 #3  │...│ EX44 #20 │     │     ║
║  │  │i5-13500  │ │i5-13500  │ │i5-13500  │   │i5-13500  │     │     ║
║  │  │64 GB     │ │64 GB     │ │64 GB     │   │64 GB     │     │     ║
║  │  │UHD770 QSV│ │UHD770 QSV│ │UHD770 QSV│   │UHD 770   │     │     ║
║  │  │~15 streams│ │~15 streams│ │~15 streams│   │~15 streams│    │     ║
║  │  │Supervisor│ │Supervisor│ │Supervisor│   │Supervisor│     │     ║
║  │  └──────────┘ └──────────┘ └──────────┘   └──────────┘     │     ║
║  │         →→→ Transcodage 4 profils ABR par stream →→→         │     ║
║  │         _ld5(500k) · _sd5(1M) · _zsd5(1.8M) · _hd5(3M)      │     ║
║  └─────────────────────────────────────────────────────────────┘     ║
║                                                                       ║
║  ┌─────────────────────┐  ┌────────────────────────────────────┐     ║
║  │   AX41-NVMe         │  │   CPX11 (Redis)                    │     ║
║  │   Monitoring        │  │   2 vCPU · 2 GB · 4.99 EUR/mois    │     ║
║  │   Prometheus        │  │   Etat orchestrateur               │     ║
║  │   Grafana           │  │   Pool workers QSV                 │     ║
║  │   AlertManager      │  │   IP : 10.0.4.1                    │     ║
║  │   intel_gpu_top     │  └────────────────────────────────────┘     ║
║  └─────────────────────┘                                             ║
╚══════════════════════════════════════════════════════════════════════╝
                    │ Internet (CDN pull depuis Edge)
╔══════════════════════════════════════════════════════════════════════╗
║                         CDN LAYER                                     ║
║                                                                       ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │        Cloudflare Geo DNS / Load Balancer                    │    ║
║  │        Geo-IP → route Afrique/Asie vers Gcore               │    ║
║  │                → route EU/USA vers Bunny                     │    ║
║  └──────────────────────┬────────────────────┬─────────────────┘    ║
║                          │                    │                      ║
║  ┌───────────────────────▼──┐    ┌────────────▼──────────────────┐  ║
║  │      Gcore CDN            │    │         Bunny CDN              │  ║
║  │  210+ PoPs worldwide      │    │     119 PoPs worldwide         │  ║
║  │  200 Tbps backbone        │    │     250 Tbps backbone          │  ║
║  │  Lagos · Nairobi ·        │    │     Paris · London ·           │  ║
║  │  Johannesburg · Abidjan   │    │     New York · Singapore       │  ║
║  │  €0.020/GB               │    │     $0.005/GB (volume >100TB)  │  ║
║  └──────────────────────────┘    └───────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════╝
                    │
     ┌──────────────▼────────────────────────────────────┐
     │                VIEWERS (Monde entier)              │
     │  iOS : MobileVLCKit 4.x + dual-player             │
     │  Android : libVLC 3.7.1 + dual-player             │
     │  HTTP-FLV · JWT éphémère (20 min) · auto-refresh  │
     │  ABR : poor→_ld5 · good→_sd5 · stable→_zsd5 ·    │
     │         excellent→_hd5                            │
     └───────────────────────────────────────────────────┘
```


---

## 1. Principes Fondamentaux


### 1.1 La règle d'or du live streaming

```
1 live = 1 flux RTMP entrant  (quelle que soit l'audience)
              |
              v
         FFmpeg encode N fois  (N qualités — invariant viewers)
              |
              v
         CDN multiplie vers M viewers  (M peut être infini)
```

Le nombre de viewers **n'impacte jamais** la charge FFmpeg ni SRS. Il impacte uniquement la bande passante CDN.

### 1.2 Choix RTMP vs SRT — Décision définitive

| Protocole | HaishinKit iOS | HaishinKit Android | Verdict |
|-----------|---------------|-------------------|---------| 
| **RTMP** | Supporte | Supporte natif | **Recommandé** |
| SRT | Supporte | Non supporté (2026) | Uniquement iOS optionnel |

### 1.3 Lazy transcoding — Le levier principal

Démarrer FFmpeg uniquement quand un **premier viewer** arrive. Stopper après **30 secondes** sans viewer.

| Maturité plateforme | Taux actifs (viewers > 0) | Streams encodés / 500 lives |
|---------------------|---------------------------|------------------------------|
| MVP beta | 20-30% | 100-150 |
| Lancement | 35-45% | 175-225 |
| **Production stable** | **60-80%** | **300-400** ← cible |

**Hypothèse retenue : 80% actifs = 400 streams encodés / 500 lives simultanés.**

### 1.4 Choix techniques justifiés

| Décision | Choix retenu | Raison |
|----------|-------------|--------|
| Transcodage | **Intel QSV (EX44)** | ASIC dédié, 4.4× moins cher que NVENC, latence minimale, stable 24h/24 |
| Codec sortie | H.264 QSV multi-profile | Compatibilité universelle · `-async_depth 1` low latency |
| **Profils ABR** | **4 profils TikTok** : `_ld5` / `_sd5` / `_zsd5` / `_hd5` | Naming TikTok/Douyin exact (reverse-engineered). Client sélectionne selon `NetworkMonitor` |
| Format delivery | HTTP-FLV | Latence 1-1.8s · parfait pour live interactif haut volume |
| LB Ingest | **HAProxy** (source hash) | Consistent hash par IP source → même streamer → même Origin node |
| CDN Afrique/Asie | **Gcore** | €0.020/GB vs Bunny $0.060/GB · 210+ PoPs avec présence Afrique |
| CDN EU/USA | **Bunny** | $0.010/GB standard · $0.005/GB volume · 119 PoPs |
| Origin | OVH Rise-1 | Réseau OVH stable, bande passante illimitée non-facturée, Anti-DDoS inclus |
| Edge | **OVH Rise-2 2 Gbps** | 2 Gbps garanti, Whitelist IP → Zero-DDoS cost, vRack inclus |
| Geo-routing | Cloudflare Geo DNS | ~$5-15/mois · Geo-steering Afrique/EU/USA sans code app |

---

## 2. Calculs de base — Capacité par serveur

### 2.1 Profils ABR (4 qualités TikTok-exact)

Naming issu du reverse-engineering des URLs CDN TikTok/Douyin :

| Profil | Résolution | Bitrate cible | Maxrate | H.264 Profile | Usage |
|--------|-----------|--------------|---------|---------------|---------|
| `_ld5` | 360p | **500 kbps** | 600k | `baseline` 3.0 | 3G faible, connexion médiocre |
| `_sd5` | 480p | **1 000 kbps** | 1 200k | `main` 3.1 | 4G standard (défaut) |
| `_zsd5` | 720p | **1 800 kbps** | 2 100k | `main` 3.2 | 4G stable |
| `_hd5` | 720p | **3 000 kbps** | 3 500k | `high` 4.0 | Wi-Fi, 5G |
| **Total max/stream** | | **6 300 kbps** | | | Tous profils actifs simultanément |

> **ABR côté client :** `NetworkMonitor.quality` (`poor` → `_ld5`, `good` → `_sd5`, `good stable` → `_zsd5`, `excellent` → `_hd5`). Switch seamless via **dual-player + `?abr_pts=XXX`** (0 freeze visible).

Capacité QSV i5-13500 à 720p H.264 `-preset veryfast` : **~15 streams actifs** par serveur (marge sécurité).

### 2.2 Capacité SRS Origin (OVH Rise-1)

- 1 Gbps port garanti → ~500 RTMP streams @ 2 Mbps entrant brut
- SRS gère jusqu'à 10 000 connexions/cœur → le Rise-1 (6 cœurs) dort
- **1 nœud Rise-1 suffit pour Phase 1 (500 lives max). 2 nœuds pour HA en Phase 2.**

### 2.3 Capacité SRS Edge (OVH Rise-2 2 Gbps)

```
Trafic par nœud Edge :
  Entrant (depuis FFmpeg Hetzner) : 600 Mbps
  Sortant (vers CDN Gcore/Bunny)  : 600 Mbps
  Total carte réseau              : 1.2 Gbps

→ Port 1 Gbps = congestion garantie
→ Port 2 Gbps garanti = marge confortable (1.2 / 2 = 60% utilisé)
→ 3 nœuds Edge = trafic divisé par 3 → chaque nœud à 400 Mbps = très safe
```

### 2.4 Securité Edge — Stratégie "Zero-DDoS Cost" (Whitelist)

Les serveurs Edge ne parlent **jamais** aux viewers directement. Ils ne parlent qu'à Gcore et BunnyCDN.

```bash
# Firewall Edge — n'autoriser QUE les IPs des CDN + les transcodeurs Hetzner
# Bloquer TOUT le reste (Internet public)

# Exemple iptables (adapter avec les vrais blocs IP Gcore/Bunny)
iptables -P INPUT DROP
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -s 10.0.2.0/24 -j ACCEPT   # Transcodeurs Hetzner (IP privées)
iptables -A INPUT -s 92.223.96.0/20 -j ACCEPT # Gcore CDN range (exemple)
iptables -A INPUT -s 185.215.64.0/22 -j ACCEPT # Bunny CDN range (exemple)
# → Résultat : DDoS de 1 Tbps = 0 impact. Le CDN absorbe tout.
# → Pas besoin de serveur anti-DDoS "GAME" à 170€/mois
```

**Économie : 3× (170€ GAME-2 → 70€ Rise-2) = 300€/mois économisés.**

### 2.5 Débit viewers par live

```
Débit moyen par viewer (distribution ABR) :
  60% choisissent 720p = 0.60 × 2.5 Mbps = 1.500 Mbps
  30% choisissent 480p = 0.30 × 1.2 Mbps = 0.360 Mbps
  10% choisissent 360p = 0.10 × 0.8 Mbps = 0.080 Mbps
  Total = 1.94 Mbps → ~0.872 GB/heure/viewer
```

| Viewers / live | GB/heure/live | GB/mois (3h/j) | Coût Bunny/mois ($0.005/GB) | Coût Gcore/mois (€0.020/GB) |
|----------------|---------------|----------------|------------------------------|-------------------------------|
| 100 | 87 GB | 7 830 GB | ~$39 | ~€157 |
| 500 | 436 GB | 39 240 GB | ~$196 | ~€785 |
| **1 000** | **872 GB** | **78 480 GB** | **~$392** | **~€1 570** |
| 60 000 (pic) | 52 320 GB | événement 2h | ~$523 (~€481/event) | ~€2 093/event |

> **Strategie regionale :** Gcore pour Afrique/Asie (~0.020 EUR/GB vs Bunny $0.060/GB Afrique = 3x moins cher).
> Bunny pour Europe/USA ($0.010/GB standard, $0.005/GB volume >100TB/mois).

### 2.6 Prix CDN verifies (avril 2026)

Sources : bunny.net/pricing, gcore.com/pricing/cdn, cloudflare.com/plans

**Bunny CDN -- Standard Network (119 PoPs, pricing par region) :**

| Region | Prix/GB |
|--------|---------|
| Europe et Amerique du Nord | $0.010/GB |
| Asie et Oceanie | $0.030/GB |
| Amerique du Sud | $0.045/GB |
| Moyen-Orient et Afrique | $0.060/GB |

**Bunny CDN -- Volume Network (10 PoPs, tarif unique global) :**

| Volume mensuel | Prix/GB |
|---------------|---------|
| < 500 TB | $0.005/GB |
| 500 TB - 1 PB | $0.004/GB |
| 1 PB - 2 PB | $0.002/GB |
| > 2 PB | Sur devis |

Minimum mensuel : $1.00. Pas de frais par requete HTTP. Transcodage Bunny Stream : gratuit.

**Gcore CDN (210+ PoPs, trafic global unifie sans surcharge regionale) :**

| | FREE | START | PRO | Enterprise |
|--|------|-------|-----|-----------|
| Prix/mois | 0 EUR | 35 EUR | 100 EUR | Sur devis |
| Trafic inclus | 1 TB | 1.5 TB | 5 TB | Sur devis |
| Overage/GB | 0.030 EUR | 0.023 EUR | 0.020 EUR | Sur devis |
| PoPs | 210+ | 210+ | 210+ | Sur devis |

Gcore Managed DNS : FREE = 0 EUR/mois (requetes illimitees, 1 health check, GeoDNS inclus). PRO = 2.49 EUR/mois (10M requetes, 5 health checks).

**Cloudflare :**

| Plan | Prix/mois | DNS | CDN | DDoS |
|------|----------|-----|-----|------|
| Free | 0 USD | Inclus | Inclus | Illimite |
| Pro | 20 USD (annuel) / 25 USD (mensuel) | Inclus | Inclus | Illimite |
| Business | 200 USD (annuel) / 250 USD (mensuel) | Inclus | Inclus | Illimite |

Cloudflare Load Balancer : 5 USD/mois (2 origins, 500K requetes incluses, 0.50 USD par 500K supplementaires).
Cloudflare Geo Steering (add-on) : +10 USD/mois. Total Geo DNS avec 2 pools : **~15 USD/mois**.

---

## 3. Phase 1 — MVP (0 → 500 lives)

### 3.1 Objectifs

- Supporter jusqu'à **500 lives simultanés**
- **400 streams actifs** réels (80% taux — scénario mature)
- **1K-2K viewers par live** sans impact infra
- HA complète : 0 single point of failure
- Monitoring + alerting opérationnels dès J1

### 3.2 Sizing exact

```
Streams actifs = 500 × 80% = 400
EX44 nécessaires = 400 / 15 = 26.6 → 27
+ 3 buffer (10%) = 30 EX44 total (safe)

Note MVP réel : taux actifs ~40% au démarrage
→ Démarrer avec 14 EX44, scaler à 30 progressivement
   (Hetzner livre les EX44 standard en quelques minutes — source : docs.hetzner.com, mars 2026)
```

> **Optimisation Phase 1 :** Commencer avec **20 EX44** (taux 60% → 300 streams). Scaler si besoin.

### 3.3 Serveurs Phase 1 — Prix reels verifies (avril 2026)

Tous les prix ci-dessous sont HT et ont ete verifies via simulation de commande sur les sites OVH et Hetzner en avril 2026.

| Role | Modele | Qte | Specs | Prix mensuel HT | Setup (une fois) | Total/mois |
|------|--------|-----|-------|-----------------|-------------------|------------|
| SRS Origin | **OVH Rise-1** | **1** | Xeon-E 2386G · 32 GB DDR4 ECC · 2x 512 GB NVMe · 1 Gbit/s illimite | **48.44 EUR** | 56.99 EUR (1er mois) | **48.44 EUR** |
| FFmpeg QSV | **Hetzner EX44** (Allemagne) | **20** | i5-13500 · 64 GB DDR4 · 2x 512 GB NVMe · 1 Gbit/s · QSV UHD 770 | **49.00 EUR** (47.30 + 1.70 IPv4) | 99.00 EUR par serveur | **980 EUR** |
| SRS Edge | **OVH Rise-2** | **3** | Xeon-E 2388G · 32 GB DDR4 ECC · 2x 512 GB NVMe · 1 Gbit/s garanti | **55.24 EUR** | 64.99 EUR (1er mois) | **165.72 EUR** |
| Option 2 Gbps public (Edge) | OVH option reseau | **3** | Upgrade bande passante 1 Gbps vers 2 Gbps garanti | **~100 EUR** | — | **~300 EUR** |
| HAProxy (x2) | **Hetzner CX33** | **2** | 4 vCPU · 8 GB · Keepalived HA | **6.99 EUR** (6.49 + 0.50 IPv4) | 0 EUR | **13.98 EUR** |
| Monitoring | Hetzner AX41-NVMe (Allemagne) | **1** | AMD Ryzen 5 3600 · 64 GB DDR4 · 2x 512 GB NVMe · Prometheus + Grafana + Loki | **44.00 EUR** (42.30 + 1.70 IPv4) | 0 EUR (promo) | **44.00 EUR** |
| Redis | Hetzner Cloud CPX11 | **1** | 2 vCPU AMD · 2 GB RAM · 40 GB SSD | **4.99 EUR** (4.49 + 0.50 IPv4) | 0 EUR | **4.99 EUR** |
| Cloudflare | Geo DNS | — | 2 origin pools | — | — | ~15 EUR |
| **Sous-total serveurs** | | | | | | **~1 569 EUR** |
| Bunny CDN EU/USA | Pay-as-you-go | — | $0.010/GB standard · $0.005/GB volume | variable | — | 200-1 000 EUR |
| Gcore CDN Afrique | PRO plan | — | 100 EUR/mois + 0.020 EUR/GB overage | variable | — | 100-400 EUR |
| **TOTAL Phase 1** | | | | | | **~1 869-2 969 EUR/mois** |

Notes importantes sur les prix :
- **OVH** : les prix affiches incluent la promo -15% en cours (avril 2026). Les prix sans promo sont Rise-1 = 56.99 EUR, Rise-2 = 64.99 EUR. Les frais d'installation sont inclus dans le montant du 1er mois.
- **Option 2 Gbps OVH** : cette option se configure dans le Control Panel OVH apres livraison du serveur, ou via l'API OVH. Elle concerne uniquement la bande passante publique. La bande passante privee (vRack) est a 10 EUR/mois pour 2 Gbps.
- **Hetzner CX33** : pas de frais d'installation. Facturation a l'heure possible (0.0104 EUR/h). 20 TB de trafic inclus, puis 1 EUR/TB.

**Hetzner EX44 -- prix detailles (verifies sur robot.hetzner.com, avril 2026) :**

Les prix HT sont ceux affiches sur le configurateur (hetzner.com/dedicated-rootserver/ex44/configurator).
Les prix TTC sont ceux affiches sur la page de commande Robot (robot.hetzner.com/order) et incluent la TVA allemande de 19%.

| | Allemagne (FSN1) HT | Allemagne (FSN1) TTC | Finlande (HEL1) HT | Finlande (HEL1) TTC |
|--|---------------------|---------------------|--------------------|--------------------|
| Serveur de base | 47.30 EUR/mois | 56.29 EUR/mois | 42.30 EUR/mois | 50.34 EUR/mois |
| Primary IPv4 | 1.70 EUR/mois | 2.02 EUR/mois | 1.70 EUR/mois | 2.02 EUR/mois |
| **Total mensuel** | **49.00 EUR/mois** | **58.31 EUR/mois** | **44.00 EUR/mois** | **52.36 EUR/mois** |
| Frais d'installation | 99.00 EUR (une fois) | 117.81 EUR (une fois) | 99.00 EUR (une fois) | 117.81 EUR (une fois) |
| **1er mois total** | **148.00 EUR** | **176.12 EUR** | **143.00 EUR** | **170.17 EUR** |
| Tarif horaire | 0.0785 EUR/h | 0.0934 EUR/h | 0.0705 EUR/h | 0.0839 EUR/h |

Pour la Finlande, le support est uniquement en anglais. Le Rescue System est en anglais.
Si tu es enregistre en tant qu'entreprise dans l'UE avec un numero de TVA intracommunautaire valide, les prix HT s'appliquent (reverse charge). Sinon, tu payes le TTC.

### 3.4 HAProxy — Load Balancer Ingest (Production (Keepalived HA))

> **Pourquoi HAProxy est conservé :**
> - Consistent hash par IP du streamer (même streamer → même SRS Origin)
> - Health checks automatiques sur les SRS Origin (API port 1985)
> - Failover < 2s avec Keepalived + VIP flottante
> - Très faible consommation — **2× Hetzner CX33 (€6.49/mois chacun)** suffisent
> - Reload sans downtime : `haproxy -sf` (0 connexion RTMP coupée)

**Déploiement : 2× Hetzner CX33** en Active/Passive avec Keepalived VRRP **unicast** (obligatoire sur cloud — le multicast ne fonctionne pas chez Hetzner).

```haproxy
# /etc/haproxy/haproxy.cfg — Version production 2026
# Déployé sur LB1 (MASTER) et LB2 (BACKUP) identiques

global
    maxconn 100000
    log 127.0.0.1 local0
    log 127.0.0.1 local1 notice
    stats socket /var/run/haproxy.sock mode 660 level admin expose-fd listeners
    user haproxy
    group haproxy
    daemon

defaults
    log global
    mode tcp
    option tcplog
    option dontlognull
    timeout connect 5s
    timeout client  3600s    # Keep-alive long pour RTMP (streams peuvent durer des heures)
    timeout server  3600s
    maxconn 100000

# --- Dashboard Stats (sécurisé) ───────────────────────────
# Accès : http://LB_IP:8404/haproxy?stats
frontend stats
    bind *:8404
    mode http
    stats enable
    stats uri /haproxy?stats
    stats refresh 10s
    stats auth admin:CHANGE_ME_STRONG_PASSWORD
    stats hide-version

# --- RTMP Ingest (port 1935) ────────────────────────────
frontend rtmp_ingest
    bind *:1935
    mode tcp
    option tcplog
    default_backend srs_origins

# --- SRT Ingest (port 10080 — iOS HaishinKit SRTHaishinKit) ────────
frontend srt_ingest
    bind *:10080
    mode tcp
    option tcplog
    default_backend srs_origins

# --- Backend SRS Origin Cluster ────────────────────────────
backend srs_origins
    mode tcp
    balance source             # Consistent hash par IP : même streamer → même Origin
    option tcp-check
    timeout connect 5s
    timeout server  3600s

    # Rate limiting RTMP : max 5 connexions simultanées par IP source
    # Empêche un bot ou script malveillant d'ouvrir des milliers de connexions
    stick-table type ip size 100k expire 30m store conn_cur
    tcp-request content track-sc0 src
    tcp-request content reject if { sc0_conn_cur ge 5 }

    # Health check sur l'API HTTP de SRS (port 1985 = SRS HTTP API)
    server srs-origin-1 10.0.1.1:1935 check port 1985 inter 2s fall 3 rise 2
    # Phase 2 : décommenter
    # server srs-origin-2 10.0.1.2:1935 check port 1985 inter 2s fall 3 rise 2
    # Phase 3 :
    # server srs-origin-3 10.0.1.3:1935 check port 1985 inter 2s fall 3 rise 2
```

**Activer `ip_nonlocal_bind` (obligatoire pour que HAProxy bind sur la VIP) :**
```bash
# Sur LB1 et LB2
echo 'net.ipv4.ip_nonlocal_bind=1' >> /etc/sysctl.conf
sysctl -p
```

**Keepalived — HA Active/Passive avec VIP flottante (unicast obligatoire sur cloud) :**
```bash
# /etc/keepalived/keepalived.conf — LB1 (MASTER)
vrrp_script chk_haproxy {
    script "/etc/keepalived/check_haproxy.sh"
    interval 2
    weight   -20
}

vrrp_instance VI_RTMP {
    state  MASTER
    interface eth0
    virtual_router_id 51
    priority 150
    advert_int 1

    # Attention : UNICAST obligatoire (multicast bloqué chez Hetzner/OVH cloud)
    unicast_src_ip  10.0.0.11   # IP privée de LB1
    unicast_peer {
        10.0.0.12               # IP privée de LB2
    }

    authentication {
        auth_type PASS
        auth_pass CHANGE_ME_VRRP_SECRET
    }

    virtual_ipaddress {
        10.0.0.10/24    # VIP — adresse donnée aux streamers HaishinKit
    }

    track_script {
        chk_haproxy
    }
}
```

```bash
# /etc/keepalived/keepalived.conf — LB2 (BACKUP)
# Même config sauf :
#   state  BACKUP
#   priority 100
#   unicast_src_ip 10.0.0.12
#   unicast_peer { 10.0.0.11 }
```

```bash
#!/bin/bash
# /etc/keepalived/check_haproxy.sh
systemctl is-active haproxy > /dev/null 2>&1
exit $?
```

**Reload HAProxy sans couper les connexions RTMP actives :**
```bash
# Génère un reload graceful — 0 connexion RTMP coupée
haproxy -f /etc/haproxy/haproxy.cfg -p /var/run/haproxy.pid -sf $(cat /var/run/haproxy.pid)
# Ou via socket (plus propre) :
echo "reload" | socat stdio /var/run/haproxy.sock
```

**Test de failover (chaos test mensuel) :**
```bash
# Sur LB1 (MASTER) : simuler une panne HAProxy
systemctl stop haproxy
# → Keepalived détecte en ~2s
# → La VIP 10.0.0.10 migre vers LB2 automatiquement
# → Les streamers se reconnectent automatiquement (<5s reconnect HaishinKit)
watch -n1 ip addr show eth0  # Surveiller sur LB2 jusqu'à voir la VIP apparaître
```

### 3.5 Configuration SRS Origin — Ingest + Forward vers FFmpeg

```nginx
# /etc/srs/srs.conf — SRS Origin (OVH Rise-1) — Production Low-Latency
listen              1935;
max_connections     5000;    # Rise-1 avec 6 cœurs gère largement 500 streamers
daemon              on;
srs_log_level       warn;
srs_log_file        ./objs/srs.log;

http_server {
    enabled     on;
    listen      8080;
}

http_api {
    enabled     on;
    listen      1985;
    crossdomain on;
}

http_hooks {
    enabled         on;
    on_publish      http://127.0.0.1:8085/api/v1/on_publish;
    on_unpublish    http://127.0.0.1:8085/api/v1/on_unpublish;
    on_play         http://127.0.0.1:8085/api/v1/on_play;
    on_stop         http://127.0.0.1:8085/api/v1/on_stop;
}

vhost __defaultVhost__ {
    # Ingest pur — SRS ne transcode pas
    # Le transcodage est délégué en totalité au cluster FFmpeg QSV (Hetzner EX44)

    # FORWARD : envoie le flux brut vers le cluster FFmpeg à la réception
    # L'orchestrateur Flask gère dynamiquement quel EX44 reçoit le flux
    forward {
        enabled     on;
        destination rtmp://127.0.0.1:9935/live/[stream];  # Loop local → orchestrateur Flask
        # En Phase 2 : rtmp://ffmpeg-cluster.internal/live/[stream]
    }

    # Webhooks de sécurité — valide le streamKey AVANT d'autoriser la publication
    http_hooks {
        enabled         on;
        on_publish      http://127.0.0.1:8085/api/v1/on_publish;
        on_unpublish    http://127.0.0.1:8085/api/v1/on_unpublish;
        on_play         http://127.0.0.1:8085/api/v1/on_play;
        on_stop         http://127.0.0.1:8085/api/v1/on_stop;
    }

    # CRITIQUE : min_latency on = force SRS à envoyer les paquets
    # immédiatement sans attendre un buffer complet
    min_latency     on;
    tcp_nodelay     on;

    play {
        gop_cache       off;    # off = pas de cache GOP → latence réduite
        queue_length    10;     # max 10 paquets en attente
        mw_latency      0;      # 0ms d'attente minimum avant envoi
        atc             off;
        atc_auto        off;
    }

    publish {
        mr          off;    # Merged-Read OFF = traitement immédiat des paquets
        mr_latency  0;      # pas de délai
    }
}
```

### 3.5 Configuration SRS Edge — HTTP-FLV output

```nginx
# /etc/srs/srs-edge.conf — SRS Edge (chaque OVH Rise-2) — Production Low-Latency
listen              1935;
max_connections     10000;   # Edge = 10 000 connexions CDN pull simultanées
daemon              on;

http_server {
    enabled         on;
    listen          8080;
}

http_api {
    enabled         on;
    listen          1985;
}

vhost __defaultVhost__ {
    cluster {
        mode            remote;
        origin          10.0.1.1:1935 10.0.1.2:1935;  # vRack OVH (privé, 0 coût)
        token_traverse  on;
    }

    http_remux {
        enabled     on;
        mount       /live/[vhost]/[app]/[stream].flv;
        hstrs       on;     # HLS-to-RTMP shim = démarrage immédiat viewer
    }

    # Paramètres low-latency identiques à l'Origin
    min_latency     on;
    tcp_nodelay     on;

    play {
        gop_cache       off;  # off = viewer rejoint le flux en direct, pas d'ancien GOP
        queue_length    10;
        mw_latency      0;    # 0ms : paquets envoyés dès réception
        atc             off;
    }

    publish {
        mr          off;
        mr_latency  0;
    }
}
```

> **Note vRack OVH :** Les flux Origin → Edge passage par le vRack privé OVH (gratuit, sécurisé, ne compte pas dans le quota public 2 Gbps). Latence <1ms entre datacenter OVH.

### 3.6 Redis — Role dans l'architecture

Redis est le seul composant qui contient un etat en temps reel. Il sert de memoire partagee entre l'orchestrateur principal et tous les EX44 workers. Sans Redis, l'orchestrateur ne saurait pas quel worker a combien de streams actifs, ni sur quel Edge chaque stream est distribue.

Ce que Redis stocke concretement :

| Cle Redis | Contenu | Exemple |
|-----------|---------|----------|
| `worker:10.0.2.1:count` | Nombre de streams actifs sur cet EX44 | `7` |
| `stream:abc123:worker` | IP du worker qui transcode ce stream | `10.0.2.1` |
| `stream:abc123:edge` | IP de l'Edge qui diffuse ce stream | `10.0.3.2` |
| `stream:abc123:viewers` | Nombre de viewers connectes | `142` |
| `stream:abc123:started_at` | Timestamp de debut du stream | `1712190061` |
| `edge:10.0.3.2:streams` | Nombre de streams sur cet Edge | `12` |
| `workers:all` | Liste de tous les EX44 enregistres | Set d'IPs |

Lorsqu'un EX44 tombe (crash), l'orchestrateur detecte via Prometheus que ce worker est `down`, decremente son compteur dans Redis, et les prochains streams sont automatiquement routing vers les autres workers. Les streams en cours sur le worker crashe sont perdus, mais le lazy transcoding les relancera des que le premier viewer se reconnecte.

Redis n'a pas besoin d'etre puissant : un CPX11 Hetzner (2 vCPU AMD, 2 GB RAM, 4.99 EUR/mois avec IPv4) suffit pour stocker l'etat de 5000 streams simultanement. En Phase 2+, on passe à un CPX32 (4 vCPU, 8 GB RAM, 14.49 EUR/mois) en cluster HA pour de la redundance. La seule contrainte est la disponibilite : c'est pourquoi on sauvegarde le dump RDB toutes les heures.

### 3.7 Orchestrateur Flask + Redis + Supervisor

```python
# /opt/livestream/orchestrator.py — Orchestrateur avec JWT + Redis
import subprocess, time, threading, logging, os
from flask import Flask, request, jsonify
import jwt, redis, xmlrpc.client

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("orchestrator")

JWT_SECRET     = os.environ["JWT_SECRET"]          # Attention : Jamais en clair — voir section Secret Management
JWT_ALGORITHM  = "HS256"
KILL_DELAY_SEC = 30
REDIS_URL      = os.environ.get("REDIS_URL", "redis://10.0.4.1:6379/0")
SUPERVISOR_USER = os.environ.get("SUPERVISOR_USER", "admin")
SUPERVISOR_PASS = os.environ["SUPERVISOR_PASS"]    # Attention : Jamais en clair

r = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)

# Pool des EX44 (Phase 1 : 20 serveurs)
WORKERS = [
    {"ip": f"10.0.2.{i}", "supervisor_port": 9001, "max_streams": 15}
    for i in range(1, 21)
]

def validate_jwt(token: str, expected_type: str, stream_key: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != expected_type: return None
        if payload.get("stream_key") != stream_key: return None
        return payload
    except Exception as e:
        logger.warning(f"JWT error: {e}")
    return None

def get_least_loaded_worker() -> dict | None:
    for w in WORKERS:
        active = int(r.get(f"worker:{w['ip']}:count") or 0)
        if active < w["max_streams"]:
            return w
    logger.error("All QSV workers at capacity!")
    return None

def get_least_loaded_edge() -> str:
    edges = ["10.0.3.1", "10.0.3.2", "10.0.3.3"]  # 3 Rise-2 Edge
    return min(edges, key=lambda e: int(r.get(f"edge:{e}:streams") or 0))

def start_ffmpeg(stream_key: str, srs_origin: str):
    worker = get_least_loaded_worker()
    if not worker:
        raise RuntimeError("No QSV worker available")
    srs_edge = get_least_loaded_edge()

    # Config Supervisor sur l'EX44
    config = f"""[program:ffmpeg-{stream_key}]
command=/opt/livestream/transcode-qsv.sh {stream_key} {srs_origin} {srs_edge}
autostart=false
autorestart=true
startretries=3
startsecs=3
stopwaitsecs=15
redirect_stderr=true
stdout_logfile=/var/log/ffmpeg/{stream_key}.log
stdout_logfile_maxbytes=10MB
"""
    proc = subprocess.run(
        ["ssh", f"root@{worker['ip']}", f"cat > /etc/supervisor/conf.d/ffmpeg-{stream_key}.conf"],
        input=config.encode(), capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Config write failed: {proc.stderr}")

    proxy = xmlrpc.client.ServerProxy(
        f"http://{SUPERVISOR_USER}:{SUPERVISOR_PASS}@{worker['ip']}:{worker['supervisor_port']}/RPC2"
    )
    proxy.supervisor.reloadConfig()
    proxy.supervisor.startProcess(f"ffmpeg-{stream_key}")

    r.incr(f"worker:{worker['ip']}:count")
    r.incr(f"edge:{srs_edge}:streams")
    r.set(f"stream:{stream_key}:worker", worker["ip"])
    r.set(f"stream:{stream_key}:edge", srs_edge)
    r.set(f"stream:{stream_key}:started_at", int(time.time()))
    logger.info(f"FFmpeg QSV started: {stream_key} on {worker['ip']} → edge {srs_edge}")

def stop_ffmpeg(stream_key: str):
    worker_ip = r.get(f"stream:{stream_key}:worker")
    srs_edge  = r.get(f"stream:{stream_key}:edge")
    if not worker_ip: return
    try:
        proxy = xmlrpc.client.ServerProxy(f"http://{SUPERVISOR_USER}:{SUPERVISOR_PASS}@{worker_ip}:9001/RPC2")
        proxy.supervisor.stopProcess(f"ffmpeg-{stream_key}")
        subprocess.run(["ssh", f"root@{worker_ip}",
            f"rm -f /etc/supervisor/conf.d/ffmpeg-{stream_key}.conf"], capture_output=True)
    except Exception as e:
        logger.error(f"Error stopping {stream_key}: {e}")
    if srs_edge: r.decr(f"edge:{srs_edge}:streams")
    r.decr(f"worker:{worker_ip}:count")
    for key in [f"stream:{stream_key}:worker", f"stream:{stream_key}:edge",
                f"stream:{stream_key}:viewers", f"stream:{stream_key}:started_at"]:
        r.delete(key)
    logger.info(f"FFmpeg stopped: {stream_key}")

@app.route("/api/v1/on_publish", methods=["POST"])
def on_publish():
    data = request.json or {}
    stream_key = data.get("stream", "")
    token = data.get("param", "").lstrip("?token=").split("&")[0]
    payload = validate_jwt(token, "stream_publish", stream_key)
    if not payload:
        return jsonify(code=403, error="Unauthorized")
    r.set(f"stream:{stream_key}:origin", data.get("server_id", "10.0.1.1"))
    logger.info(f"Stream LIVE: {stream_key}")
    return jsonify(code=0)

@app.route("/api/v1/on_unpublish", methods=["POST"])
def on_unpublish():
    stream_key = (request.json or {}).get("stream", "")
    stop_ffmpeg(stream_key)
    return jsonify(code=0)

@app.route("/api/v1/on_play", methods=["POST"])
def on_play():
    data = request.json or {}
    stream_key = data.get("stream", "")
    token = data.get("param", "").lstrip("?token=").split("&")[0]
    if not validate_jwt(token, "stream_play", stream_key):
        return jsonify(code=403, error="Unauthorized")
    r.incr(f"stream:{stream_key}:viewers")
    if not r.get(f"stream:{stream_key}:worker"):
        srs_origin = r.get(f"stream:{stream_key}:origin") or "10.0.1.1"
        try:
            start_ffmpeg(stream_key, srs_origin)
        except RuntimeError as e:
            logger.error(f"Cannot start FFmpeg: {e}")
            return jsonify(code=500, error="QSV unavailable")
    return jsonify(code=0)

@app.route("/api/v1/on_stop", methods=["POST"])
def on_stop():
    stream_key = (request.json or {}).get("stream", "")
    viewers = r.decr(f"stream:{stream_key}:viewers")
    if viewers <= 0:
        def delayed_stop():
            if int(r.get(f"stream:{stream_key}:viewers") or 0) <= 0:
                stop_ffmpeg(stream_key)
        t = threading.Timer(KILL_DELAY_SEC, delayed_stop)
        t.daemon = True; t.start()
    return jsonify(code=0)

@app.route("/metrics", methods=["GET"])
def metrics():
    active_streams = sum(1 for _ in r.scan_iter("stream:*:worker"))
    lines = [
        "# HELP active_ffmpeg_streams_total Active FFmpeg QSV jobs",
        "# TYPE active_ffmpeg_streams_total gauge",
        f"active_ffmpeg_streams_total {active_streams}",
    ]
    for w in WORKERS:
        count = int(r.get(f"worker:{w['ip']}:count") or 0)
        lines.append(f'ffmpeg_streams_per_worker{{worker="{w["ip"]}"}} {count}')
    return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8085, threaded=True)
```

### 3.7 Supervisor — Auto-recovery FFmpeg par EX44

```ini
# /etc/supervisor/conf.d/base.conf (template, copié dynamiquement par orchestrateur)
[unix_http_server]
file=/var/run/supervisor.sock

[supervisord]
logfile=/var/log/supervisor/supervisord.log
nodaemon=false

[inet_http_server]
port=*:9001
username=$SUPERVISOR_USER       # Variable d'env — ne jamais hardcoder en prod
password=$SUPERVISOR_PASS       # Variable d'env — voir section Secret Management

[include]
files=/etc/supervisor/conf.d/ffmpeg-*.conf
```

### 3.8 Optimisations TCP — Performances maximales (Origin + Edge)

```bash
#!/bin/bash
# /opt/ops/kernel-tuning.sh — À appliquer sur TOUS les serveurs (Origin + Edge + EX44)
# Sources : Linux kernel docs + production streaming best practices 2025

# --- TCP BBR (algo Google, optimal pour streaming vidéo) ──────────────────
echo "net.core.default_qdisc=fq"              >> /etc/sysctl.conf
echo "net.ipv4.tcp_congestion_control=bbr"    >> /etc/sysctl.conf

# --- Buffers réseau hauts débits (2 Gbps Edge) ────────────────────────────
echo "net.core.rmem_max=134217728"            >> /etc/sysctl.conf  # 128MB
echo "net.core.wmem_max=134217728"            >> /etc/sysctl.conf
echo "net.ipv4.tcp_rmem=4096 87380 67108864" >> /etc/sysctl.conf  # 64MB max
echo "net.ipv4.tcp_wmem=4096 65536 67108864" >> /etc/sysctl.conf
echo "net.core.netdev_max_backlog=250000"     >> /etc/sysctl.conf

# --- Connexions simultanées (5000-10000 streamers/viewers par nœud) ───────
echo "net.core.somaxconn=65535"              >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog=65535"    >> /etc/sysctl.conf
echo "net.ipv4.tcp_tw_reuse=1"              >> /etc/sysctl.conf  # reuse sockets TIME_WAIT
echo "net.ipv4.tcp_fastopen=3"              >> /etc/sysctl.conf  # TCP Fast Open
echo "net.ipv4.tcp_fin_timeout=15"          >> /etc/sysctl.conf  # libérer sockets plus vite

# --- File descriptors (chaque stream = plusieurs fds) ─────────────────────
echo "fs.file-max=2000000"                   >> /etc/sysctl.conf

sysctl -p

# --- Augmenter les limites système (ulimit) ────────────────────────────────
cat >> /etc/security/limits.conf << 'EOF'
*    soft nofile 1000000
*    hard nofile 1000000
root soft nofile 1000000
root hard nofile 1000000
EOF

# --- Si SRS/FFmpeg tournent via systemd, ajouter dans le service unit ────
# [Service]
# LimitNOFILE=1000000

echo "Kernel tuning appliqué. Redémarrer ou : sysctl -p"
```

### 3.9 Métriques Phase 1

| Métrique | Valeur cible |
|----------|-------------|
| Lives simultanés max | 500 |
| Streams actifs (80%) | 400 |
| EX44 Phase 1 | **20** (scaler si >300 streams actifs) |
| Latence live end-to-end | < 3s (HTTP-FLV) |
| Viewers max par live | **Illimité (CDN scale auto)** |
| Cold start lazy | < 5 secondes |
| Recovery FFmpeg crash | < 5s (Supervisor autorestart) |

---

## 4. Phase 2 — Scale (500 → 1 000 lives)

### 4.1 Sizing exact

```
Streams actifs = 1 000 × 80% = 800
EX44 = 800 / 15 = 53 + 5 buffer = 58 EX44
```

### 4.2 Serveurs Phase 2

| Role | Modele | Qte | Prix unit. HT | Total/mois |
|------|--------|-----|--------------|------------|
| SRS Origin | OVH Rise-1 | **2** (HA) | 48.44 EUR | 96.88 EUR |
| FFmpeg QSV | Hetzner EX44 (Allemagne) | **50** | 49.00 EUR | 2 450 EUR |
| SRS Edge | OVH Rise-2 + 2Gbps | **3** | 55.24 + ~100 = ~155 EUR | ~465 EUR |
| Monitoring | Hetzner AX41-NVMe (Allemagne) | 1 | 44.00 EUR (42.30 + 1.70 IPv4) | 44.00 EUR |
| Redis HA | Hetzner CPX32 | 2 | 14.49 EUR (13.99 + 0.50 IPv4) | 28.98 EUR |
| **Sous-total serveurs** | | | | **~3 085 EUR** |
| Bunny CDN + Gcore CDN | | -- | variable | 300-2 000 EUR |
| **TOTAL Phase 2** | | | | **~3 385-5 085 EUR** |

---

## 5. Phase 3 — Production (1 000 → 2 000 lives)

### 5.1 Sizing exact

```
Conservateur (80%) : 2 000 × 80% = 1 600 streams → 107 EX44 + 10 buffer = 117
Optimiste (60%)    : 2 000 × 60% = 1 200 streams →  80 EX44 +  8 buffer =  88

Recommandation : démarrer à 88, scaler selon métriques réelles
```

### 5.2 Serveurs Phase 3

| Rôle | Modèle | Qte | Total/mois |
|------|--------|-----|------------|
| SRS Origin | OVH Rise-1 | **2** (HA) | €130 |
| FFmpeg QSV | Hetzner EX44 | **88-117** | €4 162-5 534 |
| SRS Edge | OVH Rise-2 2Gbps | **4** | ~€420 |
| Monitoring (HA) | AX41-NVMe | 2 | €84.60 |
| Redis HA | CPX32 | 2 | €27.98 |
| **Sous-total serveurs** | | | **~€4 825-6 196** |
| CDN total | Bunny + Gcore | — | €800-5 000 |
| **TOTAL Phase 3** | | | **~€5 700-11 200** |

### 5.3 Récapitulatif scalabilité — Les 3 phases

| Phase | Lives max | Streams actifs 80% | EX44 | Total/mois (serveurs+CDN) |
|-------|----------|---------------------|------|---------------------------|
| **Phase 1** | 500 | 400 | **20** | ~€1 700-2 800 |
| **Phase 2** | 1 000 | 800 | **50** | ~€3 200-4 900 |
| **Phase 3** | 2 000 | 1 600 | **88-117** | ~€5 700-11 200 |

---

## 6. Capacité Viewers — Pourquoi le CDN est la seule limite

```
Origin  → +1 connexion RTMP (le streamer) = +0% charge
FFmpeg  → encode 1 fois les 3 qualités    = +0% (invariant viewers)
SRS Edge → +4.5 Mbps vers CDN             = négligeable sur 2G
CDN     → multiplie vers N viewers depuis ses PoPs = charge CDN

Charge Hetzner/OVH = f(lives actifs)  ← N'évolue PAS avec viewers
Charge CDN         = f(lives × viewers × bitrate)
```

### Pic 60K viewers — Anatomie

```
1 streamer → 1 slot FFmpeg QSV → SRS Edge pull une seule fois (6.3 Mbps max — 4 qualités ABR)
Bunny CDN distribue à 60 000 viewers depuis 119 PoPs
= 60 000 × 1.94 Mbps (avg _sd5 + _zsd5 mix) = 116 400 Mbps = 116 Gbps
= 116 000 / 250 000 000 Mbps (capacité Bunny) = 0.046%
```

**0 action requise côté infra Hetzner/OVH.** Le CDN absorbe automatiquement.

**Coût pic 60K viewers × 2h (Bunny $0.005/GB volume) :**
```
60 000 × 0.872 GB × 2h = 104 640 GB = ~$523 (~€481) par événement
```

---

## 7. Pourquoi garder les serveurs en Europe (pas en Afrique)

**Ne jamais acheter de serveurs dédiés en Afrique.** Voici pourquoi :
1. **Coûts prohibitifs :** Un dédié en Afrique coûte 5-15× plus cher qu'en Europe pour des specs équivalentes.
2. **Réseau instable :** Peering BGP aléatoire, coupures fréquentes, maintenabilité impossible.
3. **Gcore compense totalement :** Le backbone privé de Gcore relie ses PoPs africains (Lagos, Nairobi, Johannesburg, Abidjan…) à son réseau européen sur des fibres sous-marines ultra-rapides. La latence Gravelines→Lagos via Gcore backbone est bien inférieure à ce qu'offrirait un serveur local africain connecté sur le réseau public.

---

## 8. Monitoring & Alerting

### 8.1 Stack déployée (Hetzner AX41-NVMe)

```
Prometheus → scrape toutes les 15s
Grafana    → dashboards temps réel
AlertManager → PagerDuty + Slack
intel_gpu_top → métriques QSV par EX44
node_exporter → métriques système tous serveurs
Orchestrateur → /metrics endpoint Prometheus
```

### 8.2 Prometheus — Alertes clés

```yaml
# /etc/prometheus/alerts/livestream.yml
groups:
  - name: livestream_critical
    rules:

      - alert: QSVCapacityHigh
        expr: |
          sum(active_ffmpeg_streams_total) /
          (count(up{job="qsv-workers"}) * 15) > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Capacité QSV à {{ $value | humanizePercentage }} — commander +5 EX44"

      - alert: SRSOriginDown
        expr: up{job="srs-origin"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "SRS Origin DOWN — vérifier OVH Rise-1"

      - alert: EdgeBandwidthHigh
        expr: node_network_transmit_bytes_total{device="bond0"} / 125000000 > 1800
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "SRS Edge {{ $labels.instance }} > 1.8 Gbps sortant (limite 2G)"

      - alert: FFmpegCrashRateHigh
        expr: rate(supervisor_process_restarts_total[5m]) > 1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "FFmpeg crash rate élevé sur {{ $labels.instance }}"
```

### 8.3 Vérification QSV en temps réel

```bash
# Vérifier la charge QSV sur chaque EX44 (nécessite intel-gpu-tools)
for ip in $(seq 1 20 | xargs -I{} echo "10.0.2.{}"); do
    streams=$(ssh root@$ip "ps aux | grep -c '[f]fmpeg'" 2>/dev/null || echo "0")
    echo "$ip : $streams streams actifs"
done

# Vérifier la bande passante des Edge OVH
for ip in 10.0.3.1 10.0.3.2 10.0.3.3; do
    bw=$(ssh root@$ip "cat /sys/class/net/bond0/statistics/tx_bytes" 2>/dev/null)
    echo "Edge $ip : TX=$bw bytes"
done
```

---

## 9. FinOps — Tableau comparatif et leviers d'économie

### 9.1 Comparaison architectures (Phase 1)

| Poste | Avant (ancienne archi) | EX44 QSV + Rise-2 (actuel) | Gain/mois |
|-------|----------------------|----------------------------|-----------|
| Origin | ~€95 | **Rise-1 = €65** | -€30 |
| Edge (×3) | ~€510 | **Rise-2 2Gbps = ~€310** | **-€200** |
| FFmpeg (×20) | ~€4 246 | **EX44 QSV = €946** | **-€3 300** |
| **TOTAL infra** | **~€4 851** | **~€1 321** | **-€3 530/mois** |

> **Économie annuelle : −€42 360/an. Sur 3 ans : −€127 080.**

### 9.2 Leviers additionnels

| Action | Économie | Effort |
|--------|---------|--------|
| Lazy transcode (intégré) | -40 à -60% charge EX44 | Déjà fait |
| Bunny Volume Network (>100TB) | $0.005/GB vs $0.010 = **-50%** | Auto |
| Gcore PRO (€100/mois) | 5TB inclus, -33% vs FREE | Email |
| Hetzner Server Auction | -5 à -10% EX44 | Vérifier Robot |
| P2P CDN (Streamroot/Peer5) | -50 à -70% CDN | Phase 3 |

---

## 10. Budget consolidé — Les phases de croissance

---

### Phase 1A — Démarrage réel : 45 lives simultanés

> **Point de depart.** Infrastructure minimale mais 100% compatible avec le reste de l'archi. Tout le code, los configs, les scripts sont identiques — tu ajoutes juste des serveurs au fur et à mesure.

**Calcul :**
```
45 lives actifs ?? lazy transcoding 80% ?? 36 streams encodés simultanément
36 streams ?? 15 par EX44 ?? 3 EX44 suffisent (marge : 45 - 36 = 9 slots libres)
Bande passante Edge : 36 streams ?? 6.3 Mbps = 227 Mbps → 1 Rise-2 1 Gbps suffit
```

| Poste | Machine | Qte | Prix unitaire HT | Total/mois | Setup (une fois) |
|-------|---------|-----|-----------------|------------|------------------|
| HAProxy + SRS Origin + Orchestrateur | **OVH Rise-1** | **1** | 48.44 EUR | **48.44 EUR** | 56.99 EUR |
| FFmpeg QSV | **Hetzner EX44** (Allemagne) | **3** | 49.00 EUR (47.30 + 1.70 IPv4) | **147.00 EUR** | 3 x 99 EUR = 297 EUR |
| Hot Standby EX44 | **Hetzner EX44** (Allemagne) | **1** | 49.00 EUR | **49.00 EUR** | 99 EUR |
| SRS Edge | **OVH Rise-2** | **1** | 55.24 EUR (1 Gbps standard) | **55.24 EUR** | 64.99 EUR |
| HAProxy HA (x2) | **Hetzner CX33** | **2** | 6.99 EUR (6.49 + IPv4) | **13.98 EUR** | 0 EUR |
| Redis | Hetzner CPX11 | 1 | 4.99 EUR (4.49 + 0.50 IPv4) | **4.99 EUR** | 0 EUR |
| Monitoring | Hetzner CX33 | 1 | 6.99 EUR (6.49 + 0.50 IPv4) | **6.99 EUR** | 0 EUR |
| Cloudflare | Geo DNS | -- | -- | **~5 EUR** | -- |
| **TOTAL SERVEURS** | | | | **~330 EUR/mois** | **~518 EUR** (une fois) |
| Bunny CDN EU/USA | Variable | | $0.010/GB | **20-150 EUR** | -- |
| Gcore CDN Afrique/Asie | Variable | | 0.020 EUR/GB | **20-80 EUR** | -- |
| **TOTAL Phase 1A** | | | | **~370-560 EUR/mois** | |

A 45 lives, la bande passante Edge max est ~250 Mbps (viewers en pic), largement en dessous du 1 Gbps standard du Rise-2. L'upgrade vers 2 Gbps (~100 EUR/mois de plus) ne sera necessaire qu'au-dessus de 300 lives.

---

### Guide de Scaling à l'infini — Que commander à chaque seuil

> **Principe :** Tu regardes ton dashboard Prometheus (`active_ffmpeg_streams_total`). Dès qu'un seuil approaché à **80%**, tu commandes les ressources du palier suivant. Chaque action est indépendante, tu ajoutes UNIQUEMENT ce qui est listé.

| Seuil lives | EX44 total | Edge total | Action a faire | Cout serveurs/mois (HT) |
|-------------|-----------|-----------|----------------|------------------------|
| **45** *(Phase 1A)* | **3+1 standby** | **1x Rise-2 1Gbps** | Demarrage | ~330 EUR |
| **75** | **5+1** | 1x Rise-2 1Gbps | +2 EX44 (`add-ex44.sh`) | ~427 EUR |
| **120** | **8+1** | 1x Rise-2 1Gbps | +3 EX44 | ~574 EUR |
| **150** | **10+1** | **2x Rise-2 1Gbps** | +2 EX44 + 1 Rise-2 | ~684 EUR |
| **225** | **15+1** | 2x Rise-2 1Gbps | +5 EX44 | ~929 EUR |
| **300** *(Phase 1 MVP)* | **20+1** | **3x Rise-2 1Gbps** | +5 EX44 + 1 Rise-2 | **~1 194 EUR** |
| **375** | **25+1** | 3x Rise-2 **2Gbps** | +5 EX44 + upgrade Edge 2Gbps (+100 EUR/Edge/mois) | ~1 739 EUR |
| **500** *(Phase 2)* | **34+1** | 3x Rise-2 2Gbps | +9 EX44 + 2e SRS Origin (HA) | **~2 230 EUR** |
| **750** | **50+1** | **4x Rise-2 2Gbps** | +16 EX44 + 1 Rise-2 2Gbps | ~3 044 EUR |
| **1 000** | **67+1** | 4x Rise-2 2Gbps | +17 EX44 | ~3 877 EUR |
| **1 500** | **100+1** | **6x Rise-2 2Gbps** | +33 EX44 + 2 Rise-2 2Gbps | ~5 800 EUR |
| **2 000** *(Phase 3)* | **134+1** | **8x Rise-2 2Gbps** | +34 EX44 + 2 Rise-2 2Gbps | **~7 860 EUR** |
| **3 000** | **200+1** | 8x Rise-2 2Gbps | +66 EX44 | ~11 100 EUR |
| **5 000** | **334+1** | **12x Rise-2 2Gbps** | +134 EX44 + 4 Rise-2 2Gbps | ~18 200 EUR |
| **au-dela** | **+15/palier** | +1 Rise-2 tous les 150 lives | `add-ex44.sh` loop | N x 49.00 + M x 155.24 |

**Formule de calcul rapide à tout moment :**
```
Nombre EX44 nécessaires = ceil(lives_actifs ?? 0.80 / 15)
Nombre Edge nécessaires = ceil(EX44_total ?? 6.3 Mbps / 1 800 Mbps_par_Rise2)
   → avec 2Gbps Rise-2 dès que tu dépasses 300 lives simultanems

Ex : 500 lives × 80% = 400 actifs ÷ 15 = 27 EX44 → arrondir à 34 (marge 25%)
```

**Quand upgrader Edge de 1Gbps à 2Gbps OVH :**
```
Seuil = quand chaque Rise-2 dépasse 800 Mbps sortant vers CDN
Mesurer avec : iftop -i bond0 sur chaque Edge
Ou Grafana : node_network_transmit_bytes_total > 100 MB/s par nœud
Upgrade OVH : Control Panel → Serveur → Options réseau → Bande passante garantie
```

**Règle d'or pour ne jamais saturer :**
> Commander le palier suivant quand tu es à **70% du palier actuel** (pas 100%). Ex : si tu as 3 EX44 pour 45 lives, commande les 2 suivants à **32 lives actifs** (70% de 45).

| Poste | **Phase 1A** | Phase 1 MVP | Phase 2 | Phase 3 |
|-------|-------------|-------------|---------|--------|
| Lives max | **45** | 300 | 750 | 2 000+ |
| **HAProxy LB (2x CX33 Keepalived)** | **13.98 EUR** (x2) | **13.98 EUR** (x2) | **13.98 EUR** (x2) | **13.98 EUR** (x2) |
| SRS Origin (OVH Rise-1) | 48.44 EUR (x1) | 48.44 EUR (x1) | 96.88 EUR (x2 HA) | 96.88 EUR (x2) |
| FFmpeg EX44 QSV (49 EUR/u) | **196 EUR** (x3+1 standby) | **1 029 EUR** (x20+1) | **2 499 EUR** (x50+1) | **6 615 EUR** (x134+1) |
| SRS Edge (Rise-2) | 55.24 EUR (x1 1Gbps) | 165.72 EUR (x3 1Gbps) | ~620 EUR (x4 2Gbps) | ~1 242 EUR (x8 2Gbps) |
| Monitoring | 6.99 EUR (CX33) | 44.00 EUR (AX41) | 44.00 EUR (AX41) | 88.00 EUR (AX41 x2) |
| Redis | 4.99 EUR (CPX11) | 4.99 EUR (CPX11) | 28.98 EUR (CPX32 x2 HA) | 28.98 EUR (CPX32 x2 HA) |
| Cloudflare | ~5 EUR | ~15 EUR | ~15 EUR | ~15 EUR |
| **Sous-total serveurs** | **~330 EUR** | **~1 322 EUR** | **~3 319 EUR** | **~8 101 EUR** |
| Bunny CDN EU/USA | ~20-150 EUR | 200-1 000 EUR | 400-2 000 EUR | 800-5 000 EUR |
| Gcore CDN Afrique/Asie | ~20-80 EUR | 100-400 EUR | 200-600 EUR | 300-1 500 EUR |
| **TOTAL/mois** | **~370-560 EUR** | **~1 622-2 722 EUR** | **~3 919-5 919 EUR** | **~9 201-14 601 EUR** |

---

## 11. Annexes — Commandes Opérationnelles

### 11.1 Ajouter un EX44 au cluster

```bash
#!/bin/bash
# /opt/ops/add-ex44.sh <IP>
NEW_IP="$1"
echo "→ Installing packages on $NEW_IP..."
ssh root@$NEW_IP "apt-get update -q && apt-get install -y -q \
    supervisor ffmpeg \
    intel-media-va-driver-non-free libmfx1 vainfo libva-drm2"

echo "→ Copying scripts..."
scp /opt/livestream/transcode-qsv.sh root@$NEW_IP:/opt/livestream/
scp /etc/supervisor/supervisord.conf root@$NEW_IP:/etc/supervisor/
ssh root@$NEW_IP "mkdir -p /var/log/ffmpeg /var/run/ffmpeg"
ssh root@$NEW_IP "systemctl enable --now supervisord"

echo "→ Registering in Redis worker pool..."
redis-cli -h 10.0.4.1 SADD "workers:all" "$NEW_IP"
redis-cli -h 10.0.4.1 SET "worker:$NEW_IP:count" 0

echo "→ Testing QSV..."
ssh root@$NEW_IP "vainfo | grep H264 && echo QSV OK!"

echo "OK $NEW_IP added to cluster."
```

### 11.2 Checklist déploiement initial (Phase 1A — 45 lives)

```
[ ] OVH Rise-1 commandé + HAProxy installé (:1935 + :10080)
[ ] SRS Origin installé sur Rise-1 (:11935) + hooks vers orchestrateur :8085
[ ] Orchestrateur Flask installé + Redis CPX11 déployé (10.0.4.1)
[ ] OVH Rise-2 ×1 commandé (1 Gbps standard suffit pour Phase 1A)
[ ] vRack OVH configuré entre Rise-1 et Rise-2 (trafic privé gratuit)
[ ] Firewall Edge whitelist : seules IPs Gcore + Bunny + Hetzner workers autorisées
[ ] BBR TCP activé sur tous les serveurs OVH (sysctl -p)
[ ] Hetzner EX44 ×3 commandés (3 × 15 = 45 streams max)
[ ] Sur chaque EX44 : apt install intel-media-va-driver-non-free libmfx1 vainfo libva-drm2
[ ] usermod -aG render <ffmpeg-user> sur chaque EX44
[ ] vainfo | grep H264 → OK sur chaque EX44
[ ] ffmpeg -filters | grep vpp_qsv → OK sur chaque EX44
[ ] JWT_SECRET configuré en variable d'environnement
[ ] Bunny CDN : 1 origin pointé sur Rise-2 Edge
[ ] Gcore CDN : Pull Zone configurée pointant vers Edge IP
[ ] Cloudflare Geo DNS : Afrique/Asie → Gcore · EU/USA → Bunny
[ ] Prometheus léger : active_ffmpeg_streams_total visible
[ ] Test E2E : HaishinKit → RTMP → SRS → FFmpeg QSV → CDN → viewer OK
[ ] Test Lazy : vérifier que FFmpeg ne démarre que quand viewer arrive
[ ] Commander EX44 #4 et #5 dès que tu vois 32 lives actifs simultanés sur Grafana
```

> Phase 1A → next step :** Quand Grafana montre `active_ffmpeg_streams_total > 30`, lance immédiatement `add-ex44.sh` pour les serveurs #4 et #5. C'est transparent — le cluster s'agrandit à chaud sans coupure.

---

## 12. CI/CD Pipeline — GitHub Actions (Déploiement Production)

```yaml
# .github/workflows/deploy-infra.yml
name: Deploy Live Infrastructure

on:
  push:
    branches: [ main ]
  workflow_dispatch:

env:
  ROLLBACK_TIMEOUT: 30   # secondes avant de considérer un déploiement comme échoué

jobs:
  # ────────────────────────────────────────────────────
  # Étape 1 : Validation (lint) de toutes les configs
  # ────────────────────────────────────────────────────
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate HAProxy config
        run: |
          docker run --rm -v $PWD/haproxy:/etc/haproxy haproxy:2.9-alpine \
            haproxy -c -f /etc/haproxy/haproxy.cfg

      - name: Validate SRS Origin config
        run: |
          docker run --rm -v $PWD/srs:/etc/srs ossrs/srs:6 \
            ./objs/srs -t -c /etc/srs/srs.conf

      - name: Validate SRS Edge config
        run: |
          docker run --rm -v $PWD/srs:/etc/srs ossrs/srs:6 \
            ./objs/srs -t -c /etc/srs/srs-edge.conf

      - name: Lint transcode script (shellcheck)
        run: |
          docker run --rm -v $PWD/ffmpeg:/scripts koalaman/shellcheck:stable \
            /scripts/transcode-qsv.sh

  # ────────────────────────────────────────────────────
  # Étape 2 : Déploiement HAProxy + Keepalived (avec rollback)
  # ────────────────────────────────────────────────────
  deploy-lb:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy HAProxy + Keepalived (LB1 + LB2) avec rollback
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.LB1_IP }},${{ secrets.LB2_IP }}
          username: deploy
          key: ${{ secrets.DEPLOY_SSH_KEY }}
          script: |
            set -e
            cd /etc/haproxy
            # Sauvegarder config actuelle pour rollback
            cp haproxy.cfg haproxy.cfg.bak
            git pull
            # Valider AVANT de reload
            if ! haproxy -c -f /etc/haproxy/haproxy.cfg; then
              echo "Config HAProxy invalide — rollback"
              cp haproxy.cfg.bak haproxy.cfg
              exit 1
            fi
            # Reload graceful (0 connexion coupée)
            haproxy -f /etc/haproxy/haproxy.cfg -p /var/run/haproxy.pid -sf $(cat /var/run/haproxy.pid)
            # Vérification post-deploy : HAProxy doit répondre dans les 5s
            sleep 2
            if ! echo 'show stat' | socat stdio /var/run/haproxy.sock > /dev/null 2>&1; then
              echo "HAProxy ne répond pas — rollback"
              cp haproxy.cfg.bak haproxy.cfg
              haproxy -f /etc/haproxy/haproxy.cfg -p /var/run/haproxy.pid -sf $(cat /var/run/haproxy.pid)
              exit 1
            fi
            echo "HAProxy deploy OK"

  # ────────────────────────────────────────────────────
  # Étape 3 : Déploiement SRS Origin + Edge (avec rollback)
  # ────────────────────────────────────────────────────
  deploy-srs:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy SRS Origin (avec rollback)
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.ORIGIN_IP }}
          username: deploy
          key: ${{ secrets.DEPLOY_SSH_KEY }}
          script: |
            set -e
            cd /etc/srs
            cp srs.conf srs.conf.bak
            git pull
            # Validation config SRS avant restart
            if ! ./objs/srs -t -c /etc/srs/srs.conf; then
              echo "Config SRS invalide — rollback"
              cp srs.conf.bak srs.conf
              exit 1
            fi
            systemctl restart srs
            sleep 3
            # Vérification post-deploy : API SRS doit répondre
            if ! curl -sf http://localhost:1985/api/v1/versions > /dev/null; then
              echo "SRS ne répond pas — rollback"
              cp srs.conf.bak srs.conf
              systemctl restart srs
              exit 1
            fi
            echo "SRS Origin deploy OK"

      - name: Deploy SRS Edge (avec rollback)
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.EDGE_IPS }}    # CSV : 10.0.3.1,10.0.3.2,10.0.3.3
          username: deploy
          key: ${{ secrets.DEPLOY_SSH_KEY }}
          script: |
            set -e
            cd /etc/srs
            cp srs-edge.conf srs-edge.conf.bak
            git pull
            if ! ./objs/srs -t -c /etc/srs/srs-edge.conf; then
              cp srs-edge.conf.bak srs-edge.conf
              exit 1
            fi
            systemctl restart srs
            sleep 3
            curl -sf http://localhost:1985/api/v1/versions > /dev/null || {
              cp srs-edge.conf.bak srs-edge.conf
              systemctl restart srs
              exit 1
            }
            echo "SRS Edge deploy OK"

  # ────────────────────────────────────────────────────
  # Étape 4 : Déploiement FFmpeg Workers (graceful, sans couper les streams)
  # ────────────────────────────────────────────────────
  deploy-ffmpeg-workers:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy FFmpeg scripts to all EX44
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.WORKER_IPS }}
          username: deploy
          key: ${{ secrets.DEPLOY_SSH_KEY }}
          script: |
            set -e
            cd /opt/livestream
            cp transcode-qsv.sh transcode-qsv.sh.bak
            git pull
            # Validation shellcheck basique
            bash -n transcode-qsv.sh || {
              cp transcode-qsv.sh.bak transcode-qsv.sh
              exit 1
            }
            supervisorctl reread && supervisorctl update
            echo "FFmpeg scripts deploy OK (streams actifs non coupés)"

  # ────────────────────────────────────────────────────
  # Étape 5 : Déploiement Orchestrateur (Docker + rolling restart)
  # ────────────────────────────────────────────────────
  deploy-orchestrator:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build + deploy orchestrator Docker image
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.ORIGIN_IP }}
          username: deploy
          key: ${{ secrets.DEPLOY_SSH_KEY }}
          script: |
            set -e
            cd /opt/livestream
            git pull
            # Build nouvelle image
            docker build -t orchestrator:$(git rev-parse --short HEAD) .
            docker tag orchestrator:$(git rev-parse --short HEAD) orchestrator:latest
            # Rolling restart : démarrer le nouveau AVANT d'arrêter l'ancien
            docker stop orchestrator-old 2>/dev/null || true
            docker rename orchestrator orchestrator-old 2>/dev/null || true
            docker run -d --name orchestrator \
              --restart unless-stopped \
              --network host \
              --env-file /opt/livestream/.env \
              -v /opt/livestream/ssh:/root/.ssh:ro \
              orchestrator:latest
            sleep 3
            if ! curl -sf http://localhost:8085/metrics > /dev/null; then
              echo "Orchestrator ne répond pas — rollback"
              docker stop orchestrator && docker rm orchestrator
              docker rename orchestrator-old orchestrator
              docker start orchestrator
              exit 1
            fi
            docker rm orchestrator-old 2>/dev/null || true
            echo "Orchestrator deploy OK"
```

> **GitHub Secrets à configurer :**
> `LB1_IP`, `LB2_IP`, `ORIGIN_IP`, `EDGE_IPS`, `WORKER_IPS`, `DEPLOY_SSH_KEY` (clé ED25519 dédiée deploy)
> Ne jamais utiliser la clé root — créer un user `deploy` avec les droits nécessaires (reload HAProxy, restart SRS, supervisorctl, docker) sur chaque serveur.

---

## 13. Backup & Disaster Recovery

### 13.1 Stratégie

| Donnée | Fréquence | Destination | RTO | RPO |
|--------|----------|-------------|-----|-----|
| Config SRS (srs.conf) | Git commit | GitHub | 0 (git pull) | 0 |
| Config HAProxy | Git commit | GitHub | 0 | 0 |
| Config FFmpeg scripts | Git commit | GitHub | 0 | 0 |
| Redis dump (pool workers) | 1h | Hetzner Object Storage | 5 min | 1h |
| Orchestrateur (Flask + .env) | Quotidien | Hetzner Object Storage | 30 min | 24h |

> **Principe :** Toutes les configs sont dans Git. Un nouveau serveur = `git pull` + deploy. **Aucune config manuelle sur les serveurs.**

### 13.2 Hetzner Object Storage (S3_BUCKET) et script de backup

Hetzner Object Storage est un stockage objet compatible avec l'API S3 d'Amazon. C'est l'endroit ou les backups sont envoyes. La variable `S3_BUCKET` dans le script pointe vers un bucket que tu dois creer dans la console Hetzner (Cloud Console > Object Storage > Create Bucket).

Conformement a l'API S3, l'outil `aws s3` fonctionne avec n'importe quel provider compatible S3 via le flag `--endpoint-url`. Le endpoint `fsn1.your-objectstorage.com` est celui de Hetzner pour le datacenter de Falkenstein (FSN1) en Allemagne. Si tu utilises Helsinki (HEL1), le endpoint est `hel1.your-objectstorage.com`.

Ce qui est uploade dans ce bucket :
- Le dump RDB de Redis (contient l'etat de tous les workers et streams actifs)
- Le fichier `.env` de l'orchestrateur (contient JWT_SECRET et autres secrets)

Le bucket doit etre cree avant de lancer le script. Depuis la console Hetzner ou via leur API. L'acces se configure avec des credentials S3 (Access Key / Secret Key) generes dans Cloud Console > Security > API Tokens > Generate S3 Credentials. Ces credentials sont a ajouter dans `~/.aws/credentials` sur le serveur qui execute le backup.

```bash
#!/bin/bash
# /opt/ops/backup-daily.sh — executé via cron 0 3 * * *
set -euo pipefail

DATE=$(date +%Y%m%d-%H%M)
BACKUP_DIR="/tmp/backup-$DATE"
S3_BUCKET="s3://shopfeed-backup/live-infra"

mkdir -p "$BACKUP_DIR"

# Redis : snapshot RDB
redis-cli -h 10.0.4.1 BGSAVE
sleep 2
cp /var/lib/redis/dump.rdb "$BACKUP_DIR/redis-$DATE.rdb"

# Orchestrateur : copier .env et la BDD sqlite si utilisée
cp /opt/livestream/.env "$BACKUP_DIR/env-$DATE.bak"

# Upload vers Hetzner Object Storage (compatible S3)
aws s3 cp "$BACKUP_DIR/" "$S3_BUCKET/$DATE/" --recursive \
    --endpoint-url https://fsn1.your-objectstorage.com

# Garder 30 jours de backups seulement
aws s3 ls "$S3_BUCKET/" | sort | head -n -30 | \
    awk '{print $4}' | xargs -I{} aws s3 rm "$S3_BUCKET/{}"

echo "Backup $DATE OK"
```

### 13.3 Procédure Disaster Recovery (EX44 crash total)

```bash
# 1. Commander un nouveau EX44 chez Hetzner (livré en quelques minutes en Allemagne)
# 2. Dès livraison : lancer le script de provisioning
/opt/ops/add-ex44.sh <NOUVELLE_IP>
# Ce script installe tout (QSV drivers, FFmpeg, Supervisor) + enregistre dans Redis
# 3. Aucun stream ne se perd : le lazy transcoding relancera FFmpeg au prochain viewer
echo "Recovery complet en < 10 minutes (commande + provisioning)"
```

### 13.4 Procédure en cas de perte totale d'un datacenter

> **Scénario le plus grave :** Hetzner FSN1 ou OVH RBX/GRA totalement inaccessible.

| Composant perdu | Impact | Recovery |
|----------------|--------|----------|
| **Tous les EX44 (Hetzner)** | 0 transcodage = streamers voient le live mais pas de qualité ABR | 1. Commander 3+ EX44 dans un autre DC Hetzner (NBG1, HEL1). 2. `add-ex44.sh` sur chacun. 3. L'orchestrateur redirige automatiquement. |
| **Rise-1 Origin (OVH)** | Aucun nouveau stream accepté. Streams existants coupés. | 1. Commander Rise-1 dans un autre DC OVH. 2. `git pull` + installer SRS + orchestrateur. 3. DNS Cloudflare : mettre à jour l'IP Origin. |
| **Rise-2 Edge (OVH)** | CDN ne peut plus pull. Viewers en coupure. | 1. Commander Rise-2 dans un autre DC. 2. Installer SRS Edge. 3. Mettre à jour les origins Bunny/Gcore. |
| **Redis (Hetzner Cloud)** | Orchestrateur perd l'état des workers. Streams actifs non impactés mais pas de nouvelles allocations. | 1. Redéployer CPX11. 2. Restaurer le dump RDB depuis Object Storage. 3. Relancer l'orchestrateur. |

```bash
#!/bin/bash
# /opt/ops/dr-rebuild-from-scratch.sh
# Usage : en cas de reconstruction totale
set -euo pipefail

echo "1. Cloner le dépôt de configs..."
git clone git@github.com:shopfeed/live-infra-configs.git /tmp/live-infra

echo "2. Installer SRS Origin..."
cp /tmp/live-infra/srs/srs.conf /etc/srs/srs.conf
systemctl restart srs

echo "3. Installer orchestrateur Docker..."
cd /opt/livestream && git pull
docker build -t orchestrator:latest .
docker run -d --name orchestrator --restart unless-stopped \
  --network host --env-file /opt/livestream/.env orchestrator:latest

echo "4. Restaurer Redis..."
aws s3 cp s3://shopfeed-backup/live-infra/latest/redis-latest.rdb /var/lib/redis/dump.rdb \
  --endpoint-url https://fsn1.your-objectstorage.com
systemctl restart redis

echo "5. Provisionner les EX44..."
for ip in $(redis-cli -h 10.0.4.1 SMEMBERS workers:all); do
  /opt/ops/add-ex44.sh $ip
done

echo "Reconstruction complète."
```

---

## 14. Log Management — Loki + Promtail (Centralisé)

> **Stack :** Loki + Promtail + Grafana sur le serveur Monitoring (AX41 ou CX33).
> **Principe :** FFmpeg est très verbeux. On filtre les lignes inutiles à la source (Promtail) avant ingestion dans Loki.

### 14.1 Promtail sur chaque EX44 (FFmpeg workers)

```yaml
# /etc/promtail/promtail-config.yaml — Sur chaque EX44
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positons:
  filename: /tmp/positions.yaml

clients:
  - url: http://10.0.5.1:3100/loki/api/v1/push   # IP du serveur Monitoring

scrape_configs:
  - job_name: ffmpeg
    static_configs:
      - targets: ['localhost']
        labels:
          job: ffmpeg-transcoder
          host: __HOSTNAME__           # Remplacé par le hostname réel
          env: production
          __path__: /var/log/ffmpeg/*.log

    pipeline_stages:
      # Extraire le stream_key depuis le nom du fichier de log
      - match:
          selector: '{job="ffmpeg-transcoder"}'
          stages:
            # Filtrer les lignes trop verbeuses (debug QSV frame-level)
            - drop:
                expression: '(frame=|fps=|bitrate=.*kbits|speed=)'
            # Extraire le niveau de sévérité
            - regex:
                expression: '(?P<level>WARNING|ERROR|FATAL|INFO)'
            - labels:
                level:
```

### 14.2 Alertes Grafana critiques sur logs FFmpeg

```yaml
# Grafana Alert Rule — log pattern
expr: |
  count_over_time(
    {job="ffmpeg-transcoder"} |= "broken pipe" [5m]
  ) > 3
for: 1m
annotations:
  summary: "FFmpeg broken pipe sur {{ $labels.host }} — vérifier connexion SRS Edge"
```

```yaml
expr: |
  count_over_time(
    {job="ffmpeg-transcoder"} |= "QSV" |= "error" [5m]
  ) > 0
annotations:
  summary: "Erreur QSV sur {{ $labels.host }} — vérifier drivers intel-media-va-driver"
```

---

## 15. Security Hardening — Tous serveurs

### 15.1 SSH Hardening

```bash
#!/bin/bash
# /opt/ops/harden-ssh.sh — à lancer après chaque provisioning
# Attention : Avoir une session SSH ouverte d'abord avant d'appliquer

cat >> /etc/ssh/sshd_config << 'EOF'
PermitRootLogin no
PasswordAuthentication no
ChallengeResponseAuthentication no
UsePAM yes
X11Forwarding no
MaxAuthTries 3
AllowUsers deploy              # Seul user autorisé
Protocol 2
EOF

sshd -t && systemctl restart sshd
echo "SSH hardened OK"
```

### 15.2 Fail2Ban (tous serveurs)

```bash
apt-get install -y fail2ban

cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
ignoreip = 10.0.0.0/8 127.0.0.1/32   # IPs privées jamais bannées
bantime  = 86400                       # 24h de ban
findtime = 600
maxretry = 3

[sshd]
enabled = true
port    = ssh
filter  = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime  = 86400
EOF

systemctl enable --now fail2ban
fail2ban-client status sshd
```

### 15.3 Unattended Security Updates

```bash
apt-get install -y unattended-upgrades
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
};
Unattended-Upgrade::Automatic-Reboot "false";  # Reboot manuel contrôlé
EOF

dpkg-reconfigure -plow unattended-upgrades
```

### 15.4 Audit rapide avec Lynis (mensuel)

```bash
apt-get install -y lynis
lynis audit system --quiet --no-colors 2>&1 | grep -E '(WARNING|SUGGESTION)'
# Score cible : > 75/100
```

---

## 16. Scaling Intelligence — Le cerveau qui suit le tableau de scaling

> Avant, l'alerte se limitait a "tu es à 70%, fais quelque chose". Mais elle ne savait **pas** combien d'EX44 ajouter, si un Edge était aussi nécessaire, ni quand upgrader en 2Gbps. Maintenant, un **micro-service scaling-advisor** encode le tableau de scaling complet et envoie des les instructions exactes par Slack.

### 16.1 Architecture du système intelligent

```
Prometheus → AlertManager → Webhook → scaling-advisor.py → Slack (instructions exactes)
                                           ↓
                                    Optionnel : ordre auto via Hetzner Robot API
```

**Le scaling-advisor reçoit les métriques Prometheus et consulte le tableau de scaling encodé en dur pour déterminer EXACTEMENT le palier suivant.**

### 16.2 Le Scaling Advisor — Le cerveau (Python)

```python
#!/usr/bin/env python3
# /opt/ops/scaling-advisor.py — Micro-service qui encode le tableau de scaling
# Tourne sur le serveur Monitoring · Port 9099 · Reçoit les webhooks AlertManager
import os, json, math, requests
from flask import Flask, request, jsonify

app = Flask(__name__)
SLACK_WEBHOOK = os.environ["SLACK_WEBHOOK_URL"]


# LE TABLEAU DE SCALING — Encodé directement depuis le document
# Chaque entrée : (seuil_lives, ex44_total, edge_total, edge_bw, actions)

SCALING_TABLE = [
    {"lives":   45, "ex44":  3, "edge": 1, "edge_bw": "1Gbps", "origin": 1, "phase": "1A"},
    {"lives":   75, "ex44":  5, "edge": 1, "edge_bw": "1Gbps", "origin": 1, "phase": "1A+"},
    {"lives":  120, "ex44":  8, "edge": 1, "edge_bw": "1Gbps", "origin": 1, "phase": "1A+"},
    {"lives":  150, "ex44": 10, "edge": 2, "edge_bw": "1Gbps", "origin": 1, "phase": "1A+"},
    {"lives":  225, "ex44": 15, "edge": 2, "edge_bw": "1Gbps", "origin": 1, "phase": "1A+"},
    {"lives":  300, "ex44": 20, "edge": 3, "edge_bw": "1Gbps", "origin": 1, "phase": "1 MVP"},
    {"lives":  375, "ex44": 25, "edge": 3, "edge_bw": "2Gbps", "origin": 1, "phase": "1+"},
    {"lives":  500, "ex44": 34, "edge": 3, "edge_bw": "2Gbps", "origin": 2, "phase": "2"},
    {"lives":  750, "ex44": 50, "edge": 4, "edge_bw": "2Gbps", "origin": 2, "phase": "2+"},
    {"lives": 1000, "ex44": 67, "edge": 4, "edge_bw": "2Gbps", "origin": 2, "phase": "2+"},
    {"lives": 1500, "ex44":100, "edge": 6, "edge_bw": "2Gbps", "origin": 2, "phase": "2+"},
    {"lives": 2000, "ex44":134, "edge": 8, "edge_bw": "2Gbps", "origin": 2, "phase": "3"},
]

def get_current_tier(current_lives: int) -> dict:
    """Trouve le palier actuel dans le tableau."""
    for tier in SCALING_TABLE:
        if current_lives <= tier["lives"]:
            return tier
    # Au-delà du tableau → calcul dynamique
    return {
        "lives": current_lives,
        "ex44": math.ceil(current_lives * 0.80 / 15 * 1.25),
        "edge": math.ceil(current_lives / 150),
        "edge_bw": "2Gbps",
        "origin": 2,
        "phase": "∞"
    }

def get_next_tier(current_lives: int) -> dict:
    """Trouve le prochain palier à atteindre."""
    for i, tier in enumerate(SCALING_TABLE):
        if current_lives <= tier["lives"]:
            if i + 1 < len(SCALING_TABLE):
                return SCALING_TABLE[i + 1]
            break
    return get_current_tier(int(current_lives * 1.5))

def compute_actions(current: dict, target: dict, current_infra: dict) -> list:
    """Compare l'infra actuelle avec le palier cible et retourne les actions."""
    actions = []
    ex44_needed = target["ex44"] - current_infra.get("ex44_count", 0)
    edge_needed = target["edge"] - current_infra.get("edge_count", 0)
    origin_needed = target["origin"] - current_infra.get("origin_count", 0)
    
    if ex44_needed > 0:
        actions.append(f"Commander +{ex44_needed} EX44 (total : {target['ex44']})")
        actions.append(f"   → Lancer : for i in $(seq 1 {ex44_needed}); do /opt/ops/order-ex44.sh; done")
    
    if edge_needed > 0:
        actions.append(f"Commander +{edge_needed} OVH Rise-2 Edge")

    if target["edge_bw"] == "2Gbps" and current_infra.get("edge_bw") == "1Gbps":
        actions.append("Upgrader TOUS les Edge de 1Gbps → 2Gbps (OVH Control Panel → Options réseau)")

    if origin_needed > 0:
        actions.append(f"Commander +{origin_needed} OVH Rise-1 Origin (HA cluster)")
        actions.append("   → Décommenter srs-origin-2 dans haproxy.cfg et redéployer")

    if not actions:
        actions.append("Infra suffisante pour ce palier — rien à commander")

    return actions

@app.route("/webhook/scaling", methods=["POST"])
def scaling_webhook():
    """Reçoit les alertes AlertManager et envoie des instructions Slack exactes."""
    data = request.json or {}
    
    for alert in data.get("alerts", []):
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        
        # Extraire les métriques de l'alerte
        active_streams = float(annotations.get("active_streams", 0))
        ex44_count = int(annotations.get("ex44_count", 0))
        edge_count = int(annotations.get("edge_count", 0))
        origin_count = int(annotations.get("origin_count", 1))
        edge_bw = annotations.get("edge_bw", "1Gbps")
        
        # Calculer le nombre de lives estimés (avec lazy transcoding 80%)
        estimated_lives = int(active_streams / 0.80) if active_streams > 0 else 0
        
        current_tier = get_current_tier(estimated_lives)
        next_tier = get_next_tier(estimated_lives)
        
        current_infra = {
            "ex44_count": ex44_count,
            "edge_count": edge_count,
            "origin_count": origin_count,
            "edge_bw": edge_bw
        }
        
        actions = compute_actions(current_tier, next_tier, current_infra)
        
        # Construire le message Slack
        trigger_pct = int(active_streams / max(ex44_count * 15, 1) * 100)
        
        slack_msg = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text",
                    "text": f"SCALING ALERT — {estimated_lives} lives estimés"}},
                {"type": "section", "text": {"type": "mrkdwn",
                    "text": (
                        f"*État actuel :* {active_streams:.0f} streams actifs "
                        f"({trigger_pct}% de capacité)\n"
                        f"*Infra actuelle :* {ex44_count} EX44 · {edge_count} Edge "
                        f"({edge_bw}) · {origin_count} Origin\n"
                        f"*Palier actuel :* Phase {current_tier['phase']} "
                        f"(max {current_tier['lives']} lives)\n"
                        f"*Prochain palier :* Phase {next_tier['phase']} "
                        f"(max {next_tier['lives']} lives)"
                    )}},
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn",
                    "text": "*Actions à faire MAINTENANT :*\n" +
                            "\n".join(f"• {a}" for a in actions)}},
                {"type": "context", "elements": [{"type": "mrkdwn",
                    "text": f"Rappel : EX44 = livré en quelques minutes (Allemagne) · "
                            f"Rise-2 OVH = livraison 24-48h · "
                            f"Hot Standby EX44 absorbe le pic en attendant"}]}
            ]
        }
        
        requests.post(SLACK_WEBHOOK, json=slack_msg, timeout=10)
    
    return jsonify(status="ok")

@app.route("/scaling/status", methods=["GET"])
def scaling_status():
    """Endpoint pour vérifier l'état du scaling advisor (Prometheus health check)."""
    return jsonify(
        status="running",
        table_entries=len(SCALING_TABLE),
        max_capacity=SCALING_TABLE[-1]["lives"]
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9099)
```

### 16.3 Alertes Prometheus multi-niveaux (qui alimentent le scaling advisor)

```yaml
# /etc/prometheus/alerts/scaling-intelligence.yml
# Ces alertes envoient les métriques EXACTES au scaling-advisor via webhook

groups:
  - name: scaling_intelligence
    rules:
      # --- Alerte principale : capacité QSV à 70% ──
      - alert: ScalingNeeded_QSV
        expr: |
          sum(active_ffmpeg_streams_total) /
          (count(up{job="qsv-workers"}) * 15) > 0.70
        for: 5m
        labels:
          severity: warning
          component: qsv
        annotations:
          summary: "Cluster QSV à {{ $value | humanizePercentage }}"
          active_streams: "{{ with query \"sum(active_ffmpeg_streams_total)\" }}{{ . | first | value }}{{ end }}"
          ex44_count: "{{ with query \"count(up{job='qsv-workers'})\" }}{{ . | first | value }}{{ end }}"
          edge_count: "{{ with query \"count(up{job='srs-edge'})\" }}{{ . | first | value }}{{ end }}"
          origin_count: "{{ with query \"count(up{job='srs-origin'})\" }}{{ . | first | value }}{{ end }}"

      # --- Alerte critique : Hot Standby en cours d'utilisation ──
      - alert: HotStandbyActivated
        expr: |
          min(ffmpeg_streams_per_worker) == 0
          and sum(active_ffmpeg_streams_total) /
          (count(up{job="qsv-workers"}) * 15) > 0.85
        for: 2m
        labels:
          severity: critical
          component: hot_standby
        annotations:
          summary: "Le Hot Standby EX44 est maintenant actif — commander immédiatement !"

      # --- Alerte bande passante Edge : upgrade vers 2Gbps nécessaire ──
      - alert: EdgeBandwidthHigh
        expr: |
          max(rate(node_network_transmit_bytes_total{job="srs-edge",device="bond0"}[5m])) * 8
          > 800000000
        for: 10m
        labels:
          severity: warning
          component: edge_bandwidth
        annotations:
          summary: "SRS Edge {{ $labels.instance }} à {{ $value | humanize }}bps — upgrader vers 2Gbps"

      # --- Alerte Origin : max_connections approché ──
      - alert: OriginConnectionsHigh
        expr: |
          srs_connections_total{job="srs-origin"} /
          srs_max_connections{job="srs-origin"} > 0.70
        for: 5m
        labels:
          severity: warning
          component: origin
        annotations:
          summary: "SRS Origin à {{ $value | humanizePercentage }} de connexions max — ajouter un 2e Origin"
```

### 16.4 AlertManager — Route vers le scaling-advisor

```yaml
# /etc/alertmanager/alertmanager.yml
route:
  group_by: ['alertname', 'component']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h
  receiver: 'slack-default'
  routes:
    # Toutes les alertes scaling → vers le webhook intelligent
    - match:
        alertname: ScalingNeeded_QSV
      receiver: 'scaling-advisor'
    - match:
        alertname: HotStandbyActivated
      receiver: 'scaling-advisor-critical'
    - match:
        alertname: EdgeBandwidthHigh
      receiver: 'scaling-advisor'
    - match:
        alertname: OriginConnectionsHigh
      receiver: 'scaling-advisor'

receivers:
  - name: 'slack-default'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#live-infra'
        send_resolved: true

  - name: 'scaling-advisor'
    webhook_configs:
      - url: 'http://localhost:9099/webhook/scaling'
        send_resolved: false

  - name: 'scaling-advisor-critical'
    webhook_configs:
      - url: 'http://localhost:9099/webhook/scaling'
        send_resolved: false
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#live-infra-urgent'
        send_resolved: true
        title: 'HOT STANDBY ACTIVÉ — COMMANDER MAINTENANT'
```

### 16.5 Script Hetzner Robot API — Commande automatique

```bash
#!/bin/bash
# /opt/ops/order-ex44.sh — Commande un ou plusieurs EX44 via Hetzner Robot API
# Doc API : https://robot.hetzner.com/doc/webservice/en.html
# Usage : ./order-ex44.sh [nombre]    (défaut : 1)
#
# Attention : EX44 standard en Allemagne = livré en quelques minutes (source : docs.hetzner.com mars 2026)
# Customisé (hardware additionnel) = 2-3 jours ouvrés. On commande TOUJOURS standard.
# Le add-ex44.sh sera lancé automatiquement après commande.

set -euo pipefail
source /opt/livestream/.env

COUNT=${1:-1}
PRODUCT="EX44"
LOCATION="FSN1"

for i in $(seq 1 $COUNT); do
  RESPONSE=$(curl -s -X POST \
    -u "$HETZNER_ROBOT_USER:$HETZNER_ROBOT_PASS" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "product=$PRODUCT&location=$LOCATION&authorized_key[]=SSH_KEY_FINGERPRINT" \
    https://robot-ws.your-server.de/order/server/transaction)

  ORDER_ID=$(echo "$RESPONSE" | jq -r '.transaction.id')
  echo "[$i/$COUNT] EX44 commandé — Order ID : $ORDER_ID"

  curl -s -X POST "$SLACK_WEBHOOK_URL" \
    -H 'Content-type: application/json' \
    --data "{\"text\": \"EX44 #$i/$COUNT commandé (ID: $ORDER_ID). Livré en quelques minutes (Allemagne).\"}"
done
```

### 16.6 Overflow physique — Le "Hot Standby" EX44

> **Pourquoi on ne fait PAS d'overflow sur Cloud VM (CCX33) :** Les instances cloud n'ont pas de puce graphique (Intel QSV). Elles transcoderaient via CPU (`libx264`) et s'effondreraient à 3-5 streams maximum avec une qualité instable. Inacceptable en production.

**La vraie solution 100% robuste : un EX44 de secours dédié.**
Pour seulement **49.00 EUR/mois**, on maintient en permanence **1 serveur EX44 de plus que nécessaire**, configuré et connecté au cluster, mais gardé vide (priorité basse dans l'orchestrateur Redis).

* **Capacité immédiate :** 15-20 streams instantanés de marge en cas de pic inattendu.
* **Le temps tampon :** Ce serveur encaisse l'overflow instantanément pendant que le nouveau EX44 est commandé et provisionné (~10-15 minutes total en Allemagne).
* **Alerte dédiée :** Dès qu'un seul stream est routé sur le Hot Standby, l'alerte `HotStandbyActivated` (severity: critical) se déclenche et le scaling-advisor envoie immédiatement les instructions exactes sur Slack.

### 16.7 Comment ça fonctionne concrètement (scénario réel)

```
Situation : Tu as 3 EX44 (Phase 1A, 45 lives max). 32 lives actifs.

1. Prometheus mesure : active_ffmpeg_streams_total = 26 (32 × 80% lazy)
2. Calcul : 26 / (3 × 15) = 57.8% → pas encore 70% → rien

Situation : 38 lives actifs

1. Prometheus mesure : active_ffmpeg_streams_total = 30
2. Calcul : 30 / (3 × 15) = 66.7% → pas encore 70% → rien

Situation : 42 lives actifs

1. Prometheus mesure : active_ffmpeg_streams_total = 34
2. Calcul : 34 / (3 × 15) = 75.5% → > 70% pendant 5 min → ALERTE !
3. AlertManager envoie au scaling-advisor (port 9099)
4. Le scaling-advisor consulte le tableau :
   - Lives estimés : 34 / 0.80 = 42
   - Palier actuel : 45 (Phase 1A, 3 EX44)
   - Palier suivant : 75 (Phase 1A+, 5 EX44)
5. Actions calculées : "+2 EX44"
6. Message Slack envoyé :

   SCALING ALERT — 42 lives estimés
   ─────────────────────────────────
   État actuel : 34 streams actifs (75% de capacité)
   Infra actuelle : 3 EX44 · 1 Edge (1Gbps) · 1 Origin
   Palier actuel : Phase 1A (max 45 lives)
   Prochain palier : Phase 1A+ (max 75 lives)
   ─────────────────────────────────
   Actions à faire MAINTENANT :
   • Commander +2 EX44 (total : 5)
   •    → Lancer : for i in $(seq 1 2); do /opt/ops/order-ex44.sh; done
   ─────────────────────────────────
   Hot Standby absorbe le pic en attendant (15 streams de marge)
```

---

## 17. Cost Monitoring & Budget Alerts

```yaml
# /etc/prometheus/alerts/costs.yml
groups:
  - name: cost_alerts
    rules:
      # Alert si trop de serveurs QSV idle (gaspillage)
      - alert: QSVWorkersUnderutilized
        expr: |
          sum(active_ffmpeg_streams_total) /
          (count(up{job="qsv-workers"}) * 15) < 0.20
        for: 60m
        annotations:
          summary: "Cluster QSV à moins de 20% d'utilisation depuis 1h — retirer des EX44 ?"

      # Alert bande passante Edge anormalement haute
      - alert: EdgeBandwidthAnomaly
        expr: |
          node_network_transmit_bytes_total{device="bond0"} / 125000000 > 1500
        for: 10m
        annotations:
          summary: "SRS Edge {{ $labels.instance }} à {{ $value }} Mbps — pic de trafic ou attaque ?"
```

**Dashboard Grafana — Indicateurs financiers en temps réel :**
```promql
# Coût Bunny estimé en temps réel (total bytes sortant Edge * prix)
(sum(increase(node_network_transmit_bytes_total{job="srs-edge"}[1h])) / 1e9) * 0.005
# → Afficher en panel "€/heure CDN estimé"

# Nombre d'EX44 actifs vs nécessaires
count(up{job="qsv-workers"}) - ceil(sum(active_ffmpeg_streams_total) / 15 * 1.25)
# → Positif = surplus (gaspillage), négatif = manque (urgence commande)
```

---

## 18. Secret Management — Production

> **Règle #1 :** Aucun mot de passe, clé JWT, ou secret ne doit être en clair dans un fichier Git ou dans une config commitée. Utiliser `os.environ[]` partout (déjà fait dans l'orchestrateur) + fichier `.env` exclu de Git.

### 18.1 Approche .env + sops (Phase 1A — Simple & Efficace)

```bash
# /opt/livestream/.env — CE FICHIER NE DOIT JAMAIS ÊTRE DANS GIT
# chmod 600 /opt/livestream/.env && chown deploy:deploy /opt/livestream/.env

JWT_SECRET=votre-secret-jwt-genere-avec-openssl-rand-hex-64
SUPERVISOR_USER=admin
SUPERVISOR_PASS=votre-mot-de-passe-supervisor-fort
REDIS_URL=redis://10.0.4.1:6379/0
HETZNER_ROBOT_USER=xxx
HETZNER_ROBOT_PASS=xxx
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
```

```bash
# Générer des secrets forts :
openssl rand -hex 64   # JWT_SECRET
openssl rand -base64 24  # SUPERVISOR_PASS
```

```bash
# Protection du fichier .env :
chmod 600 /opt/livestream/.env
chown deploy:deploy /opt/livestream/.env
echo '.env' >> /opt/livestream/.gitignore
```

### 18.2 Chiffrement des secrets avec sops + age (recommandé)

```bash
# Installer sops + age
apt-get install -y age
wget -O /usr/local/bin/sops https://github.com/getsops/sops/releases/latest/download/sops-v3.9.0.linux.amd64
chmod +x /usr/local/bin/sops

# Générer une clé age (une seule fois)
age-keygen -o /opt/livestream/age-key.txt   # → Garder cette clé en lieu sûr

# Chiffrer le .env
sops --age $(cat /opt/livestream/age-key.txt | grep 'public key' | awk '{print $4}') \
  -e /opt/livestream/.env > /opt/livestream/.env.enc

# .env.enc peut être commité dans Git → seul le serveur avec age-key.txt peut le déchiffrer
# Déchiffrer :
SOPS_AGE_KEY_FILE=/opt/livestream/age-key.txt sops -d /opt/livestream/.env.enc > /opt/livestream/.env
```

### 18.3 Migration vers HashiCorp Vault (Phase 2+)

> **Pour Phase 2+ à 50+ serveurs :** Remplacer `.env` par HashiCorp Vault en mode agent. Chaque serveur obtient ses secrets via un token auto-renewé. Non nécessaire pour Phase 1A.

---

## 19. Observabilité — Latence End-to-End Viewer

> La métrique la plus critique pour la qualité perçue : **combien de temps entre le moment où le viewer se connecte et la première frame affichée.**

### 19.1 Métrique côté client (HaishinKit / VLC player)

```swift
// iOS — MobileVLCKit : mesurer le Time-To-First-Frame (TTFF)
let startTime = CFAbsoluteTimeGetCurrent()
player.play()

// Dès que le premier frame apparaît :
func mediaPlayerStateChanged(_ notification: Notification) {
    if player.isPlaying {
        let ttff = CFAbsoluteTimeGetCurrent() - startTime
        // Envoyer la métrique à votre backend analytics
        Analytics.track("live_ttff", ["ttff_ms": Int(ttff * 1000), "quality": currentQuality])
    }
}
// Cible : TTFF < 1.5 secondes en 4G · < 800ms en WiFi
```

### 19.2 Métrique côté serveur (Prometheus + orchestrateur)

```python
# Ajouter dans orchestrator.py — endpoint /metrics
# Latence d'allocation : temps entre on_play et FFmpeg started
allocation_latency = []

@app.route("/api/v1/on_play", methods=["POST"])
def on_play():
    start = time.time()
    # ... logique existante ...
    latency = time.time() - start
    allocation_latency.append(latency)
    return jsonify(code=0)

# Dans /metrics :
lines.append(f"ffmpeg_allocation_latency_seconds {sum(allocation_latency[-100:])/max(len(allocation_latency[-100:]),1)}")
```

```yaml
# Prometheus alert : TTFF serveur trop lent
- alert: FFmpegAllocationSlow
  expr: ffmpeg_allocation_latency_seconds > 2
  for: 1m
  annotations:
    summary: "Allocation FFmpeg trop lente ({{ $value }}s) — cluster QSV surchargé ?"
```

### 19.3 Dashboard Grafana — Qualité Viewer

```promql
# Panels recommandés :
# 1. TTFF moyen (envoyé par le client iOS/Android)
histogram_quantile(0.95, rate(live_ttff_seconds_bucket[5m]))

# 2. Latence allocation FFmpeg (serveur)
ffmpeg_allocation_latency_seconds

# 3. Nombre de viewers par qualité (ABR distribution)
sum by (quality) (live_viewers_by_quality)

# 4. Taux d'erreur viewer (connexions échouées / total)
rate(live_connection_errors_total[5m]) / rate(live_connection_total[5m])
```

---

## 20. IPv6 — Activation sur tous les serveurs

> **Pourquoi en 2026 :** En Afrique, l'espace IPv4 est saturé. Beaucoup d'opérateurs mobiles (MTN, Orange, Airtel) utilisent IPv6-only ou dual-stack. Ne pas avoir IPv6 = perdre une partie des viewers africains.

### 20.1 Activation (tous serveurs Hetzner + OVH)

```bash
#!/bin/bash
# /opt/ops/enable-ipv6.sh
# Hetzner assigne un /64 par défaut — OVH aussi dans le vRack

# Vérifier l'adresse IPv6 assignée
ip -6 addr show scope global

# Si pas d'IPv6 visible, vérifier :
cat /etc/netplan/01-netcfg.yaml
# Doit contenir :
#   addresses:
#     - 2a01:xxxx:xxxx::1/64
#   gateway6: fe80::1

# Activer IPv6 forwarding (pour HAProxy)
echo 'net.ipv6.conf.all.forwarding=1' >> /etc/sysctl.conf
sysctl -p

# Tester
ping6 -c3 google.com && echo "IPv6 OK"
```

### 20.2 HAProxy IPv6 (ajouter dans haproxy.cfg)

```haproxy
# Ajouter des bind IPv6 dans chaque frontend
frontend rtmp_ingest
    bind *:1935           # IPv4
    bind :::1935          # IPv6
    mode tcp
    default_backend srs_origins

frontend srt_ingest
    bind *:10080
    bind :::10080         # IPv6
    mode tcp
    default_backend srs_origins
```

### 20.3 Cloudflare + CDN IPv6

> Cloudflare active IPv6 par défaut sur toutes les zones. Bunny CDN et Gcore CDN supportent IPv6 nativement sur tous leurs PoPs. **Rien à configurer côté CDN.**

---

## 21. Containerisation Orchestrateur — Docker + systemd

> **Pourquoi Dockeriser l'orchestrateur Flask :** Faciliter les mises à jour, les rollbacks, l'isolation des dépendances, et assurer un restart automatique via `--restart unless-stopped`.

### 21.1 Dockerfile

```dockerfile
# /opt/livestream/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Dépendances système SSH (pour commander les EX44 Supervisor via xmlrpc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator.py .
COPY transcode-qsv.sh .

EXPOSE 8085

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD curl -sf http://localhost:8085/metrics || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8085", "--workers", "4", "--threads", "2", "orchestrator:app"]
```

### 21.2 requirements.txt

```
flask==3.1.*
PyJWT==2.9.*
redis==5.2.*
gunicorn==23.*
```

### 21.3 Déploiement Docker + systemd

```bash
# Build et lancement
cd /opt/livestream
docker build -t orchestrator:latest .

docker run -d \
  --name orchestrator \
  --restart unless-stopped \
  --network host \
  --env-file /opt/livestream/.env \
  -v /opt/livestream/ssh:/root/.ssh:ro \
  orchestrator:latest

# Vérifier :
curl -s http://localhost:8085/metrics
```

> **Rollback en cas de problème :** Le CI/CD garde l'ancien container sous le nom `orchestrator-old`. Rollback en 5 secondes :
```bash
docker stop orchestrator && docker rm orchestrator
docker rename orchestrator-old orchestrator
docker start orchestrator
```

---

## 22. Cloudflare WAF — Protection contre les bots

> Même avec le firewall whitelist sur les Edge, il faut protéger les endpoints **publics** (les URLs CDN que reçoivent les viewers).

### 22.1 Règles WAF Cloudflare (Free Plan)

```
Règle 1 — Rate limit viewers (via Cloudflare Rules > Rate Limiting)
  IF : URI path contains "/live/"
  AND : Requests per 10 seconds > 30 per IP
  THEN : Block for 60 seconds
  REASON : Un viewer légitime ne fait qu'1 requête (le stream HTTP-FLV est continu)

Règle 2 — Block non-browser user agents
  IF : URI path contains "/live/"
  AND : User Agent does NOT contain ("VLC" OR "MobileVLCKit" OR "libVLC" OR "Lavf" OR "stagefright")
  THEN : Challenge (CAPTCHA)
  REASON : Seuls nos players natifs et les CDN doivent accéder aux flux

Règle 3 — Geo-block pays non ciblés (optionnel)
  IF : Country NOT IN (fr, us, gb, de, ng, ci, sn, cm, cd, ke, za, ...)
  THEN : Block
  REASON : Réduire le trafic indésirable depuis les pays où on n'opère pas
```

### 22.2 Rate Limiting SRS Edge (natif)

```nginx
# Ajouter dans srs-edge.conf vhost
vhost __defaultVhost__ {
    # ... configs existantes ...

    # Limiter le débit par IP viewer (anti-abuse)
    # SRS ne supporte pas le rate-limiting natif sur HTTP-FLV
    # → Solution : s'appuyer sur HAProxy stats + Cloudflare WAF
    # → Les Edge sont protégés par le firewall whitelist (seuls CDN IPs accèdent)
}
```

> **Double protection :** Le firewall whitelist empêche tout accès direct aux Edge. Le WAF Cloudflare protège les viewers qui passent par le CDN. Les deux ensemble = zéro surface d'attaque.

---

*Document version 4.4 -- Architecture Production -- Intel QSV EX44 -- Avril 2026*

*Prix verifies via checkout et configurateur avril 2026 (HT sauf mention) :*
*Hetzner EX44 Allemagne : 49.00 EUR/mois (47.30 + 1.70 IPv4) -- setup 99 EUR -- TTC : 58.31 EUR/mois*
*Hetzner EX44 Finlande : 44.00 EUR/mois (42.30 + 1.70 IPv4) -- setup 99 EUR -- TTC : 52.36 EUR/mois*
*Hetzner AX41-NVMe Allemagne : 44.00 EUR/mois (42.30 + 1.70 IPv4) -- setup 0 EUR (promo)*
*Hetzner AX41-NVMe Finlande : 38.40 EUR/mois (36.70 + 1.70 IPv4) -- setup 0 EUR (promo)*
*Hetzner CX33 Cloud : 6.99 EUR/mois (6.49 + 0.50 IPv4) -- pas de setup*
*Hetzner CPX11 Cloud : 4.99 EUR/mois (4.49 + 0.50 IPv4) -- 2 vCPU AMD, 2 GB RAM*
*Hetzner CPX32 Cloud : 14.49 EUR/mois (13.99 + 0.50 IPv4) -- 4 vCPU AMD, 8 GB RAM*
*OVH Rise-1 : 48.44 EUR/mois HT (promo -15%, base 56.99) -- setup inclus 1er mois*
*OVH Rise-2 : 55.24 EUR/mois HT (promo -15%, base 64.99) -- setup inclus 1er mois*
*OVH option 2 Gbps public : ~100 EUR HT/mois supplementaire -- vRack 2 Gbps : 10 EUR/mois*
*Bunny CDN : $0.010/GB EU standard -- $0.005/GB volume -- $0.060/GB Afrique*
*Gcore CDN : PRO 100 EUR/mois -- 5TB inclus -- 0.020 EUR/GB overage -- 210+ PoPs*
*Cloudflare : LB 5 USD/mois + Geo Steering 10 USD/mois = ~15 USD/mois*
*Sources : hetzner.com -- eco.ovhcloud.com -- bunny.net/pricing -- gcore.com/pricing -- cloudflare.com/plans*

