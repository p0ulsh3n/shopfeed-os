# Live Streaming Infrastructure -- ShopFeed OS

**Architecture de Production -- Hetzner + OVH + CDN -- Phase 1 a Phase 3**

> Stack : HaishinKit → RTMP/SRT → HAProxy → SRS Origin (OVH Rise-1) → FFmpeg Intel QSV (Hetzner EX44) → SRS Edge (OVH Rise-2) → Gcore CDN + Bunny CDN → Viewers
> Version : 4.4 -- Avril 2026

---

## Table des matieres

1. [Vue d'ensemble](#1-vue-densemble)
2. [Intel QSV -- Transcodage materiel](#2-intel-qsv----transcodage-materiel)
3. [Schemas d'architecture](#3-schemas-darchitecture)
4. [Calculs de capacite](#4-calculs-de-capacite)
5. [Configuration des composants](#5-configuration-des-composants)
6. [Securite et hardening](#6-securite-et-hardening)
7. [Monitoring et observabilite](#7-monitoring-et-observabilite)
8. [Scaling intelligent](#8-scaling-intelligent)
9. [Budgets par phase](#9-budgets-par-phase)
10. [Operations](#10-operations)
11. [Annexes](#11-annexes)

---

## 1. Vue d'ensemble

### 1.1 Technologies et references

| Composant | Technologie | Lien officiel |
|-----------|------------|---------------|
| **Media Server** | SRS (Simple Realtime Server) v6.x | https://ossrs.net |
| **Ingest Mobile iOS** | HaishinKit 2.2.5 (RTMP + SRT) | https://github.com/HaishinKit/HaishinKit.swift |
| **Ingest Mobile Android** | HaishinKit.kt (RTMP) | https://github.com/HaishinKit/HaishinKit.kt |
| **Transcodage GPU** | FFmpeg + Intel QSV (h264_qsv) | https://ffmpeg.org/ffmpeg-codecs.html#QSV-Encoders |
| **Load Balancer** | HAProxy | https://www.haproxy.org |
| **CDN EU/USA** | Bunny CDN | https://bunny.net |
| **CDN Afrique/Asie** | Gcore CDN | https://gcore.com/cdn |
| **Serveurs transcodage** | Hetzner EX44 (i5-13500 QSV) | https://www.hetzner.com/dedicated-rootserver/ex44 |
| **Serveur Origin** | OVH Rise-1 | https://eco.ovhcloud.com/fr/rise/rise-1/ |
| **Serveurs Edge** | OVH Rise-2 + 2 Gbps | https://eco.ovhcloud.com/fr/rise/rise-2/ |
| **Geo DNS** | Cloudflare Load Balancer | https://developers.cloudflare.com/load-balancing |

> **SRS (Simple Realtime Server)** : Serveur RTMP/HTTP-FLV/WebRTC open-source, equivalent cloud-grade developpe par une equipe chinoise (ex-Alibaba). Meme stack utilise par Taobao Live. Version 6.x stable recommandee.

### 1.2 Principes fondamentaux

**La regle d'or du live streaming :**

```
1 live = 1 flux RTMP entrant  (quelle que soit l'audience)
              |
              v
         FFmpeg encode N fois  (N qualites -- invariant viewers)
              |
              v
         CDN multiplie vers M viewers  (M peut etre infini)
```

Le nombre de viewers n'impacte jamais la charge FFmpeg ni SRS. Il impacte uniquement la bande passante CDN.

**RTMP vs SRT :**

| Protocole | HaishinKit iOS | HaishinKit Android | Verdict |
|-----------|---------------|-------------------|---------| 
| **RTMP** | Supporte | Supporte natif | **Recommande** |
| SRT | Supporte | Non supporte (2026) | Uniquement iOS optionnel |

**Lazy transcoding (levier principal) :** Demarrer FFmpeg uniquement quand un premier viewer arrive. Stopper apres 30 secondes sans viewer.

| Maturite plateforme | Taux actifs (viewers > 0) | Streams encodes / 500 lives |
|---------------------|---------------------------|------------------------------|
| MVP beta | 20-30% | 100-150 |
| Lancement | 35-45% | 175-225 |
| **Production stable** | **60-80%** | **300-400** |

Hypothese retenue : 80% actifs = 400 streams encodes / 500 lives simultanes.

### 1.3 Choix techniques justifies

| Decision | Choix retenu | Raison |
|----------|-------------|--------|
| Transcodage | **Intel QSV (EX44)** | ASIC dedie, 4.4x moins cher que NVENC, latence minimale, stable 24h/24 |
| Codec sortie | H.264 QSV multi-profile | Compatibilite universelle, `-async_depth 1` low latency |
| **Profils ABR** | **4 profils TikTok** : `_ld5` / `_sd5` / `_zsd5` / `_hd5` | Naming TikTok/Douyin exact (reverse-engineered). Client selectionne selon `NetworkMonitor` |
| Format delivery | HTTP-FLV | Latence 1-1.8s, parfait pour live interactif haut volume |
| LB Ingest | **HAProxy** (source hash) | Consistent hash par IP source, meme streamer vers meme Origin node |
| CDN Afrique/Asie | **Gcore** | 0.020 EUR/GB vs Bunny $0.060/GB, 210+ PoPs avec presence Afrique |
| CDN EU/USA | **Bunny** | $0.010/GB standard, $0.005/GB volume, 119 PoPs |
| Origin | OVH Rise-1 | Reseau OVH stable, bande passante illimitee non-facturee, Anti-DDoS inclus |
| Edge | **OVH Rise-2 2 Gbps** | 2 Gbps garanti, Whitelist IP, Zero-DDoS cost, vRack inclus |
| Geo-routing | Cloudflare Geo DNS | ~$5-15/mois, Geo-steering Afrique/EU/USA sans code app |

### 1.4 Pourquoi garder les serveurs en Europe (pas en Afrique)

Ne jamais acheter de serveurs dedies en Afrique :

1. **Couts prohibitifs :** Un dedie en Afrique coute 5-15x plus cher qu'en Europe pour des specs equivalentes.
2. **Reseau instable :** Peering BGP aleatoire, coupures frequentes, maintenabilite impossible.
3. **Gcore compense totalement :** Le backbone prive de Gcore relie ses PoPs africains (Lagos, Nairobi, Johannesburg, Abidjan) a son reseau europeen sur des fibres sous-marines ultra-rapides. La latence Gravelines-Lagos via Gcore backbone est bien inferieure a ce qu'offrirait un serveur local africain connecte sur le reseau public.

---

## 2. Intel QSV -- Transcodage materiel

### 2.1 Capacites UHD 770

L'Intel i5-13500 possede l'**UHD Graphics 770**, avec **2 Multi-Format Codec Engines (MFX) independants**. Ce n'est pas une "iGPU" grand-public, c'est un ASIC (Application-Specific Integrated Circuit) dedie au transcodage video.

> **Capacite hardware confirmee 2026 :** L'UHD 770 peut encoder jusqu'a 15-20 flux 720p simultanement en production. Le hardware est theoriquement capable de 1080p mais il n'est pas utilise en production (voir note ci-dessous). Tests montrent des vitesses 8x-10x la vitesse reelle pour un flux single.

> **Note 1080p :** Le 1080p n'est PAS dans les profils de production. Les 4 profils (`_ld5` 360p / `_sd5` 480p / `_zsd5` 720p / `_hd5` 720p) couvrent 100% des usages mobiles. Le 1080p sera envisage uniquement si un marche mature le justifie (il demanderait ~5-6 Mbps/stream et reduirait la capacite par EX44 a ~10 streams actifs).

- **2 MFX Engines independants :** Chaque moteur gere sa propre file de streams. Les 2 ensemble sur l'UHD 770 permettent le pipeline multi-stream sans contention.
- **Performance reelle :** 1 serveur EX44 = 15 a 20 flux complets `_ld5`+`_sd5`+`_zsd5`+`_hd5` simultanes, CPU a <10%
- **Low Latency absolu :** `-async_depth 1` + `-look_ahead 0` reduisent le delai RTMP-FLV au minimum
- **Stabilite thermique :** Le moteur QSV integre ne "throttle" jamais sous charge 24h/24 continue
- **Cout :** 49.00 EUR/mois (47.30 + 1.70 IPv4) + 99 EUR setup, soit ~3.3 EUR/stream/mois (15 streams actifs)
- **AV1 :** decodage OK, encodage NON. L'i5-13500 decode l'AV1 hardware mais ne peut pas l'encoder. On reste sur H.264 QSV pour l'output live.

### 2.2 Drivers Linux requis

```bash
# Ubuntu 22.04/24.04 -- OBLIGATOIRE sur chaque EX44 avant de lancer FFmpeg
# Sans ces 2 packages, FFmpeg ne detecte pas QSV et tombera en soft CPU = catastrophe
apt-get update && apt-get install -y \
    intel-media-va-driver-non-free \
    libmfx1 \
    vainfo \
    libva-drm2

# Ajouter l'utilisateur ffmpeg au groupe render (acces /dev/dri sans root)
usermod -aG render $(whoami)

# Verifier que les 2 MFX Engines sont detectes :
vainfo | grep H264
# Doit retourner : VAProfileH264Main, VAProfileH264High, VAProfileH264ConstrainedBaseline

# Verifier que h264_qsv est disponible dans votre build FFmpeg :
ffmpeg -encoders 2>/dev/null | grep qsv
# Doit lister : h264_qsv, hevc_qsv, vp9_qsv...

# Verifier que vpp_qsv (filtre critique pour split multi-output) est dispo :
ffmpeg -filters 2>/dev/null | grep qsv
# Doit lister : vpp_qsv, scale_qsv

# Noyau Linux 6.x minimum (EX44 livre avec Ubuntu 24.04 par defaut chez Hetzner)
uname -r  # Doit retourner 6.x
```

### 2.3 Commande FFmpeg QSV -- Production

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

**Notes critiques QSV :**
- **4 profils TikTok-exact** : `_ld5` / `_sd5` / `_zsd5` / `_hd5` (naming reverse-engineered depuis les URLs CDN de TikTok/Douyin)
- `profile:v baseline/main/high` : niveaux H.264 corrects selon la qualite (compatibilite appareils bas de gamme pour `_ld5`)
- `vpp_qsv` dans `filter_complex` : CORRECT. `scale_qsv` ne fonctionne pas avec `split=4` multi-output
- `-hwaccel_output_format qsv` : frames decodes restent en memoire GPU (aucune copie vers RAM)
- `-look_ahead 0` : CRITIQUE. Elimine 500ms-2s de latence buffer
- `-g 30` (1 keyframe/s a 30fps) : standard live. Le viewer rejoint le stream a moins d'1s d'attente
- `-async_depth 1` : latence pipeline minimale, pour live interactif fluide

---

## 3. Schemas d'architecture

### 3.1 Schema 1 -- Flux de donnees complet (Data Flow)

```
┌───────┐
│                         STREAMERS (Mobiles)                             │
│                                                                         │
│   ┌──────────────────────┐         ┌──────────────────────┐            │
│   │   iOS (HaishinKit    │         │  Android (HaishinKit │            │
│   │   2.2.5 swift)       │         │  .kt)                │            │
│   │   SRT (priorite)     │         │  RTMP uniquement     │            │
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
               │  → selectionne EX44 disponible  │
               │  → spawn FFmpeg sur le worker   │
               └────────────────┬───────────────┘
                                │ Dispatch RTMP → worker
                                ▼
     ┌──────────────────────────────────────────────────────┐
     │              FFmpeg QSV Cluster                       │
     │              20x Hetzner EX44                        │
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
                                │ RTMP 4x qualites
                                │ (via vRack Hetzner → OVH)
                                ▼
     ┌──────────────────────────────────────────────────────┐
     │           SRS Edge Cluster (ossrs.net)                │
     │           3x OVH Rise-2                              │
     │           Xeon-E 2388G · 64 GB · 2 Gbps garanti      │
     │                                                       │
     │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
     │  │   Edge #1    │ │   Edge #2    │ │   Edge #3    │ │
     │  │ 2 Gbps out   │ │ 2 Gbps out   │ │ 2 Gbps out   │ │
     │  │ HTTP-FLV :80 │ │ HTTP-FLV :80 │ │ HTTP-FLV :80 │ │
     │  └──────────────┘ └──────────────┘ └──────────────┘ │
     │                                                       │
     │  Firewall WHITELIST : seuls Gcore + Bunny IPs     │
     │     autorises en entree — DDoS = 0 impact            │
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
          │  0.020 EUR/GB   │  │    $0.005/GB (vol)   │
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

### 3.2 Schema 2 -- Infrastructure physique et reseau

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
║                               │ vRack OVH (prive, gratuit)           ║
║  ┌──────────────────────────────────────────────────────────────┐    ║
║  │          3x OVH Rise-2  (SRS Edge Cluster)                   │    ║
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
║  │              20x Hetzner EX44 — FFmpeg Cluster              │     ║
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
║  │  0.020 EUR/GB             │    │     $0.005/GB (volume >100TB)  │  ║
║  └──────────────────────────┘    └───────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════╝
                    │
     ┌──────────────▼────────────────────────────────────┐
     │                VIEWERS (Monde entier)              │
     │  iOS : MobileVLCKit 4.x + dual-player             │
     │  Android : libVLC 3.7.1 + dual-player             │
     │  HTTP-FLV · JWT ephemere (20 min) · auto-refresh  │
     │  ABR : poor→_ld5 · good→_sd5 · stable→_zsd5 ·    │
     │         excellent→_hd5                            │
     └───────────────────────────────────────────────────┘
```

---

## 4. Calculs de capacite

### 4.1 Profils ABR (4 qualites TikTok-exact)

Naming issu du reverse-engineering des URLs CDN TikTok/Douyin :

| Profil | Resolution | Bitrate cible | Maxrate | H.264 Profile | Usage |
|--------|-----------|--------------|---------|---------------|---------| 
| `_ld5` | 360p | **500 kbps** | 600k | `baseline` 3.0 | 3G faible, connexion mediocre |
| `_sd5` | 480p | **1 000 kbps** | 1 200k | `main` 3.1 | 4G standard (defaut) |
| `_zsd5` | 720p | **1 800 kbps** | 2 100k | `main` 3.2 | 4G stable |
| `_hd5` | 720p | **3 000 kbps** | 3 500k | `high` 4.0 | Wi-Fi, 5G |
| **Total max/stream** | | **6 300 kbps** | | | Tous profils actifs simultanement |

> **ABR cote client :** `NetworkMonitor.quality` (`poor` → `_ld5`, `good` → `_sd5`, `good stable` → `_zsd5`, `excellent` → `_hd5`). Switch seamless via dual-player + `?abr_pts=XXX` (0 freeze visible).

Capacite QSV i5-13500 a 720p H.264 `-preset veryfast` : **~15 streams actifs** par serveur (marge securite).

### 4.2 Capacite SRS Origin (OVH Rise-1)

- 1 Gbps port garanti → ~500 RTMP streams @ 2 Mbps entrant brut
- SRS gere jusqu'a 10 000 connexions/coeur → le Rise-1 (6 coeurs) dort
- 1 noeud Rise-1 suffit pour Phase 1 (500 lives max). 2 noeuds pour HA en Phase 2.

### 4.3 Capacite SRS Edge (OVH Rise-2 2 Gbps)

```
Trafic par noeud Edge :
  Entrant (depuis FFmpeg Hetzner) : 600 Mbps
  Sortant (vers CDN Gcore/Bunny)  : 600 Mbps
  Total carte reseau              : 1.2 Gbps

→ Port 1 Gbps = congestion garantie
→ Port 2 Gbps garanti = marge confortable (1.2 / 2 = 60% utilise)
→ 3 noeuds Edge = trafic divise par 3 → chaque noeud a 400 Mbps = tres safe
```

### 4.4 Strategie Zero-DDoS Cost (Whitelist Edge)

Les serveurs Edge ne parlent jamais aux viewers directement. Ils ne parlent qu'a Gcore et BunnyCDN.

```bash
# Firewall Edge -- n'autoriser QUE les IPs des CDN + les transcodeurs Hetzner
# Bloquer TOUT le reste (Internet public)

# Exemple iptables (adapter avec les vrais blocs IP Gcore/Bunny)
iptables -P INPUT DROP
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -s 10.0.2.0/24 -j ACCEPT   # Transcodeurs Hetzner (IP privees)
iptables -A INPUT -s 92.223.96.0/20 -j ACCEPT # Gcore CDN range (exemple)
iptables -A INPUT -s 185.215.64.0/22 -j ACCEPT # Bunny CDN range (exemple)
# Resultat : DDoS de 1 Tbps = 0 impact. Le CDN absorbe tout.
# Pas besoin de serveur anti-DDoS "GAME" a 170 EUR/mois
```

Economie : 3x (170 EUR GAME-2 → 70 EUR Rise-2) = 300 EUR/mois economises.

### 4.5 Debit viewers par live

```
Debit moyen par viewer (distribution ABR) :
  60% choisissent 720p = 0.60 x 2.5 Mbps = 1.500 Mbps
  30% choisissent 480p = 0.30 x 1.2 Mbps = 0.360 Mbps
  10% choisissent 360p = 0.10 x 0.8 Mbps = 0.080 Mbps
  Total = 1.94 Mbps → ~0.872 GB/heure/viewer
```

| Viewers / live | GB/heure/live | GB/mois (3h/j) | Cout Bunny/mois ($0.005/GB) | Cout Gcore/mois (0.020 EUR/GB) |
|----------------|---------------|----------------|------------------------------|-------------------------------|
| 100 | 87 GB | 7 830 GB | ~$39 | ~157 EUR |
| 500 | 436 GB | 39 240 GB | ~$196 | ~785 EUR |
| **1 000** | **872 GB** | **78 480 GB** | **~$392** | **~1 570 EUR** |
| 60 000 (pic) | 52 320 GB | evenement 2h | ~$523 (~481 EUR/event) | ~2 093 EUR/event |

> **Strategie regionale :** Gcore pour Afrique/Asie (~0.020 EUR/GB vs Bunny $0.060/GB Afrique = 3x moins cher). Bunny pour Europe/USA ($0.010/GB standard, $0.005/GB volume >100TB/mois).

### 4.6 Capacite Viewers -- Pourquoi le CDN est la seule limite

```
Origin  → +1 connexion RTMP (le streamer) = +0% charge
FFmpeg  → encode 1 fois les 3 qualites    = +0% (invariant viewers)
SRS Edge → +4.5 Mbps vers CDN             = negligeable sur 2G
CDN     → multiplie vers N viewers depuis ses PoPs = charge CDN

Charge Hetzner/OVH = f(lives actifs)  ← N'evolue PAS avec viewers
Charge CDN         = f(lives x viewers x bitrate)
```

**Pic 60K viewers -- Anatomie :**

```
1 streamer → 1 slot FFmpeg QSV → SRS Edge pull une seule fois (6.3 Mbps max)
Bunny CDN distribue a 60 000 viewers depuis 119 PoPs
= 60 000 x 1.94 Mbps = 116 Gbps
= 116 000 / 250 000 000 Mbps (capacite Bunny) = 0.046%
```

0 action requise cote infra Hetzner/OVH. Le CDN absorbe automatiquement.

**Cout pic 60K viewers x 2h (Bunny $0.005/GB volume) :**
```
60 000 x 0.872 GB x 2h = 104 640 GB = ~$523 (~481 EUR) par evenement
```

### 4.7 Prix CDN verifies (avril 2026)

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

Minimum mensuel : $1.00. Pas de frais par requete HTTP.

**Gcore CDN (210+ PoPs, trafic global unifie) :**

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

### 4.8 Prix serveurs verifies (avril 2026)

Tous les prix ci-dessous sont HT et ont ete verifies via simulation de commande sur les sites OVH et Hetzner en avril 2026.

**Hetzner EX44 -- Detail par localisation :**

| | Allemagne (FSN1) HT | Allemagne (FSN1) TTC | Finlande (HEL1) HT | Finlande (HEL1) TTC |
|--|---------------------|---------------------|--------------------|--------------------| 
| Serveur de base | 47.30 EUR/mois | 56.29 EUR/mois | 42.30 EUR/mois | 50.34 EUR/mois |
| Primary IPv4 | 1.70 EUR/mois | 2.02 EUR/mois | 1.70 EUR/mois | 2.02 EUR/mois |
| **Total mensuel** | **49.00 EUR/mois** | **58.31 EUR/mois** | **44.00 EUR/mois** | **52.36 EUR/mois** |
| Frais d'installation | 99.00 EUR (une fois) | 117.81 EUR (une fois) | 99.00 EUR (une fois) | 117.81 EUR (une fois) |
| **1er mois total** | **148.00 EUR** | **176.12 EUR** | **143.00 EUR** | **170.17 EUR** |
| Tarif horaire | 0.0785 EUR/h | 0.0934 EUR/h | 0.0705 EUR/h | 0.0839 EUR/h |

Pour la Finlande, le support est uniquement en anglais. Si l'entreprise est enregistree dans l'UE avec un numero de TVA intracommunautaire valide, les prix HT s'appliquent (reverse charge). Sinon, le TTC s'applique.

**Tous les serveurs -- Recapitulatif :**

| Modele | Specs | Prix HT/mois | Setup | Usage |
|--------|-------|-------------|-------|-------|
| **Hetzner EX44** (Allemagne) | i5-13500, 64 GB, 2x512 NVMe, QSV UHD 770 | 49.00 EUR (47.30 + 1.70 IPv4) | 99.00 EUR | FFmpeg QSV workers |
| **Hetzner AX41-NVMe** (Allemagne) | Ryzen 5 3600, 64 GB, 2x512 NVMe | 44.00 EUR (42.30 + 1.70 IPv4) | 0 EUR (promo) | Monitoring |
| **Hetzner CX33** Cloud | 4 vCPU Intel, 8 GB RAM, 80 GB SSD | 6.99 EUR (6.49 + 0.50 IPv4) | 0 EUR | HAProxy, Monitoring P1A |
| **Hetzner CPX11** Cloud | 2 vCPU AMD, 2 GB RAM, 40 GB SSD | 4.99 EUR (4.49 + 0.50 IPv4) | 0 EUR | Redis P1 |
| **Hetzner CPX32** Cloud | 4 vCPU AMD, 8 GB RAM, 160 GB SSD | 14.49 EUR (13.99 + 0.50 IPv4) | 0 EUR | Redis HA P2+ |
| **OVH Rise-1** | Xeon-E 2386G, 32 GB ECC, 2x512 NVMe, 1 Gbps | 48.44 EUR (promo -15%) | 56.99 EUR (1er mois) | SRS Origin |
| **OVH Rise-2** | Xeon-E 2388G, 32 GB ECC, 2x512 NVMe, 1 Gbps | 55.24 EUR (promo -15%) | 64.99 EUR (1er mois) | SRS Edge |
| **OVH option 2 Gbps** | Bande passante publique garantie | ~100 EUR/mois | -- | Edge upgrade |

Notes :
- **OVH** : les prix incluent la promo -15% en cours (avril 2026). Les frais d'installation sont inclus dans le montant du 1er mois.
- **Option 2 Gbps OVH** : se configure dans le Control Panel OVH apres livraison du serveur. Concerne uniquement la bande passante publique. La bande passante privee (vRack) est a 10 EUR/mois pour 2 Gbps.
- **Hetzner CX33** : facturation a l'heure possible (0.0104 EUR/h). 20 TB de trafic inclus, puis 1 EUR/TB.

---

## 5. Configuration des composants

### 5.1 HAProxy -- Load Balancer Ingest (Keepalived HA)

> **Pourquoi HAProxy :**
> - Consistent hash par IP du streamer (meme streamer vers meme SRS Origin)
> - Health checks automatiques sur les SRS Origin (API port 1985)
> - Failover < 2s avec Keepalived + VIP flottante
> - Faible consommation -- 2x Hetzner CX33 (6.99 EUR/mois chacun) suffisent
> - Reload sans downtime : `haproxy -sf` (0 connexion RTMP coupee)

Deploiement : 2x Hetzner CX33 en Active/Passive avec Keepalived VRRP **unicast** (obligatoire sur cloud -- le multicast ne fonctionne pas chez Hetzner).

```haproxy
# /etc/haproxy/haproxy.cfg -- Version production 2026
# Deploye sur LB1 (MASTER) et LB2 (BACKUP) identiques

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

# --- Dashboard Stats (securise) ---
frontend stats
    bind *:8404
    mode http
    stats enable
    stats uri /haproxy?stats
    stats refresh 10s
    stats auth admin:CHANGE_ME_STRONG_PASSWORD
    stats hide-version

# --- RTMP Ingest (port 1935) ---
frontend rtmp_ingest
    bind *:1935
    mode tcp
    option tcplog
    default_backend srs_origins

# --- SRT Ingest (port 10080 -- iOS HaishinKit) ---
frontend srt_ingest
    bind *:10080
    mode tcp
    option tcplog
    default_backend srs_origins

# --- Backend SRS Origin Cluster ---
backend srs_origins
    mode tcp
    balance source             # Consistent hash par IP : meme streamer vers meme Origin
    option tcp-check
    timeout connect 5s
    timeout server  3600s

    # Rate limiting RTMP : max 5 connexions simultanees par IP source
    stick-table type ip size 100k expire 30m store conn_cur
    tcp-request content track-sc0 src
    tcp-request content reject if { sc0_conn_cur ge 5 }

    # Health check sur l'API HTTP de SRS (port 1985 = SRS HTTP API)
    server srs-origin-1 10.0.1.1:1935 check port 1985 inter 2s fall 3 rise 2
    # Phase 2 : decommenter
    # server srs-origin-2 10.0.1.2:1935 check port 1985 inter 2s fall 3 rise 2
```

**Activer `ip_nonlocal_bind` :**
```bash
# Sur LB1 et LB2 (obligatoire pour que HAProxy bind sur la VIP)
echo 'net.ipv4.ip_nonlocal_bind=1' >> /etc/sysctl.conf
sysctl -p
```

**Keepalived -- HA Active/Passive (unicast obligatoire sur cloud) :**
```bash
# /etc/keepalived/keepalived.conf -- LB1 (MASTER)
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

    # UNICAST obligatoire (multicast bloque chez Hetzner/OVH cloud)
    unicast_src_ip  10.0.0.11   # IP privee de LB1
    unicast_peer {
        10.0.0.12               # IP privee de LB2
    }

    authentication {
        auth_type PASS
        auth_pass CHANGE_ME_VRRP_SECRET
    }

    virtual_ipaddress {
        10.0.0.10/24    # VIP -- adresse donnee aux streamers HaishinKit
    }

    track_script {
        chk_haproxy
    }
}
```

```bash
# /etc/keepalived/keepalived.conf -- LB2 (BACKUP)
# Meme config sauf :
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
haproxy -f /etc/haproxy/haproxy.cfg -p /var/run/haproxy.pid -sf $(cat /var/run/haproxy.pid)
# Ou via socket :
echo "reload" | socat stdio /var/run/haproxy.sock
```

**Test de failover (chaos test mensuel) :**
```bash
# Sur LB1 (MASTER) : simuler une panne HAProxy
systemctl stop haproxy
# Keepalived detecte en ~2s
# La VIP 10.0.0.10 migre vers LB2 automatiquement
# Les streamers se reconnectent automatiquement (<5s reconnect HaishinKit)
watch -n1 ip addr show eth0  # Surveiller sur LB2 jusqu'a voir la VIP apparaitre
```

### 5.2 SRS Origin -- Ingest + Forward

```nginx
# /etc/srs/srs.conf -- SRS Origin (OVH Rise-1) -- Production Low-Latency
listen              1935;
max_connections     5000;    # Rise-1 avec 6 coeurs gere largement 500 streamers
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
    forward {
        enabled     on;
        destination rtmp://127.0.0.1:9935/live/[stream];
    }

    http_hooks {
        enabled         on;
        on_publish      http://127.0.0.1:8085/api/v1/on_publish;
        on_unpublish    http://127.0.0.1:8085/api/v1/on_unpublish;
        on_play         http://127.0.0.1:8085/api/v1/on_play;
        on_stop         http://127.0.0.1:8085/api/v1/on_stop;
    }

    min_latency     on;
    tcp_nodelay     on;

    play {
        gop_cache       off;
        queue_length    10;
        mw_latency      0;
        atc             off;
        atc_auto        off;
    }

    publish {
        mr          off;
        mr_latency  0;
    }
}
```

### 5.3 SRS Edge -- HTTP-FLV output

```nginx
# /etc/srs/srs-edge.conf -- SRS Edge (chaque OVH Rise-2) -- Production Low-Latency
listen              1935;
max_connections     10000;
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
        origin          10.0.1.1:1935 10.0.1.2:1935;  # vRack OVH (prive, 0 cout)
        token_traverse  on;
    }

    http_remux {
        enabled     on;
        mount       /live/[vhost]/[app]/[stream].flv;
        hstrs       on;
    }

    min_latency     on;
    tcp_nodelay     on;

    play {
        gop_cache       off;
        queue_length    10;
        mw_latency      0;
        atc             off;
    }

    publish {
        mr          off;
        mr_latency  0;
    }
}
```

> **Note vRack OVH :** Les flux Origin vers Edge passent par le vRack prive OVH (gratuit, securise, ne compte pas dans le quota public 2 Gbps). Latence <1ms entre datacenter OVH.

### 5.4 Redis -- Role dans l'architecture

Redis est le seul composant qui contient un etat en temps reel. Il sert de memoire partagee entre l'orchestrateur principal et tous les EX44 workers. Sans Redis, l'orchestrateur ne saurait pas quel worker a combien de streams actifs, ni sur quel Edge chaque stream est distribue.

**Ce que Redis stocke :**

| Cle Redis | Contenu | Exemple |
|-----------|---------|----------|
| `worker:10.0.2.1:count` | Nombre de streams actifs sur cet EX44 | `7` |
| `stream:abc123:worker` | IP du worker qui transcode ce stream | `10.0.2.1` |
| `stream:abc123:edge` | IP de l'Edge qui diffuse ce stream | `10.0.3.2` |
| `stream:abc123:viewers` | Nombre de viewers connectes | `142` |
| `stream:abc123:started_at` | Timestamp de debut du stream | `1712190061` |
| `edge:10.0.3.2:streams` | Nombre de streams sur cet Edge | `12` |
| `workers:all` | Liste de tous les EX44 enregistres | Set d'IPs |

Lorsqu'un EX44 tombe (crash), l'orchestrateur detecte via Prometheus que ce worker est `down`, decremente son compteur dans Redis, et les prochains streams sont automatiquement routes vers les autres workers. Les streams en cours sur le worker crashe sont perdus, mais le lazy transcoding les relancera des que le premier viewer se reconnecte.

Redis n'a pas besoin d'etre puissant : un CPX11 Hetzner (2 vCPU AMD, 2 GB RAM, 4.99 EUR/mois avec IPv4) suffit pour stocker l'etat de 5000 streams simultanement. En Phase 2+, on passe a un CPX32 (4 vCPU, 8 GB RAM, 14.49 EUR/mois) en cluster HA pour de la redundance. La seule contrainte est la disponibilite : c'est pourquoi on sauvegarde le dump RDB toutes les heures.

### 5.5 Orchestrateur Flask + Redis + Supervisor

```python
# /opt/livestream/orchestrator.py -- Orchestrateur avec JWT + Redis
import subprocess, time, threading, logging, os
from flask import Flask, request, jsonify
import jwt, redis, xmlrpc.client

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("orchestrator")

JWT_SECRET     = os.environ["JWT_SECRET"]
JWT_ALGORITHM  = "HS256"
KILL_DELAY_SEC = 30
REDIS_URL      = os.environ.get("REDIS_URL", "redis://10.0.4.1:6379/0")
SUPERVISOR_USER = os.environ.get("SUPERVISOR_USER", "admin")
SUPERVISOR_PASS = os.environ["SUPERVISOR_PASS"]

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
    edges = ["10.0.3.1", "10.0.3.2", "10.0.3.3"]
    return min(edges, key=lambda e: int(r.get(f"edge:{e}:streams") or 0))

def start_ffmpeg(stream_key: str, srs_origin: str):
    worker = get_least_loaded_worker()
    if not worker:
        raise RuntimeError("No QSV worker available")
    srs_edge = get_least_loaded_edge()

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
    logger.info(f"FFmpeg QSV started: {stream_key} on {worker['ip']} -> edge {srs_edge}")

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

### 5.6 Supervisor -- Auto-recovery FFmpeg par EX44

```ini
# /etc/supervisor/conf.d/base.conf (template, copie dynamiquement par orchestrateur)
[unix_http_server]
file=/var/run/supervisor.sock

[supervisord]
logfile=/var/log/supervisor/supervisord.log
nodaemon=false

[inet_http_server]
port=*:9001
username=$SUPERVISOR_USER
password=$SUPERVISOR_PASS

[include]
files=/etc/supervisor/conf.d/ffmpeg-*.conf
```

### 5.7 Optimisations TCP -- Performances maximales

```bash
#!/bin/bash
# /opt/ops/kernel-tuning.sh -- A appliquer sur TOUS les serveurs (Origin + Edge + EX44)

# --- TCP BBR (algo Google, optimal pour streaming video) ---
echo "net.core.default_qdisc=fq"              >> /etc/sysctl.conf
echo "net.ipv4.tcp_congestion_control=bbr"    >> /etc/sysctl.conf

# --- Buffers reseau hauts debits (2 Gbps Edge) ---
echo "net.core.rmem_max=134217728"            >> /etc/sysctl.conf  # 128MB
echo "net.core.wmem_max=134217728"            >> /etc/sysctl.conf
echo "net.ipv4.tcp_rmem=4096 87380 67108864" >> /etc/sysctl.conf  # 64MB max
echo "net.ipv4.tcp_wmem=4096 65536 67108864" >> /etc/sysctl.conf
echo "net.core.netdev_max_backlog=250000"     >> /etc/sysctl.conf

# --- Connexions simultanees (5000-10000 streamers/viewers par noeud) ---
echo "net.core.somaxconn=65535"              >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog=65535"    >> /etc/sysctl.conf
echo "net.ipv4.tcp_tw_reuse=1"              >> /etc/sysctl.conf
echo "net.ipv4.tcp_fastopen=3"              >> /etc/sysctl.conf
echo "net.ipv4.tcp_fin_timeout=15"          >> /etc/sysctl.conf

# --- File descriptors ---
echo "fs.file-max=2000000"                   >> /etc/sysctl.conf

sysctl -p

# --- Augmenter les limites systeme (ulimit) ---
cat >> /etc/security/limits.conf << 'EOF'
*    soft nofile 1000000
*    hard nofile 1000000
root soft nofile 1000000
root hard nofile 1000000
EOF

echo "Kernel tuning applique. Redemarrer ou : sysctl -p"
```

### 5.8 Containerisation Orchestrateur -- Docker

```dockerfile
# /opt/livestream/Dockerfile
FROM python:3.12-slim

WORKDIR /app

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

```
# /opt/livestream/requirements.txt
flask==3.1.*
PyJWT==2.9.*
redis==5.2.*
gunicorn==23.*
```

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

# Verifier :
curl -s http://localhost:8085/metrics
```

**Rollback en cas de probleme :**
```bash
docker stop orchestrator && docker rm orchestrator
docker rename orchestrator-old orchestrator
docker start orchestrator
```

---

## 6. Securite et hardening

### 6.1 SSH Hardening

```bash
#!/bin/bash
# /opt/ops/harden-ssh.sh -- a lancer apres chaque provisioning

cat >> /etc/ssh/sshd_config << 'EOF'
PermitRootLogin no
PasswordAuthentication no
ChallengeResponseAuthentication no
UsePAM yes
X11Forwarding no
MaxAuthTries 3
AllowUsers deploy
Protocol 2
EOF

sshd -t && systemctl restart sshd
echo "SSH hardened OK"
```

### 6.2 Fail2Ban

```bash
apt-get install -y fail2ban

cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
ignoreip = 10.0.0.0/8 127.0.0.1/32
bantime  = 86400
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
```

### 6.3 Unattended Security Updates

```bash
apt-get install -y unattended-upgrades
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
};
Unattended-Upgrade::Automatic-Reboot "false";
EOF

dpkg-reconfigure -plow unattended-upgrades
```

### 6.4 Audit Lynis (mensuel)

```bash
apt-get install -y lynis
lynis audit system --quiet --no-colors 2>&1 | grep -E '(WARNING|SUGGESTION)'
# Score cible : > 75/100
```

### 6.5 Secret Management

> Aucun mot de passe, cle JWT, ou secret ne doit etre en clair dans un fichier Git ou dans une config commitee. Utiliser `os.environ[]` partout + fichier `.env` exclu de Git.

**Approche .env (Phase 1A) :**

```bash
# /opt/livestream/.env -- NE DOIT JAMAIS ETRE DANS GIT
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
# Generer des secrets forts :
openssl rand -hex 64   # JWT_SECRET
openssl rand -base64 24  # SUPERVISOR_PASS

# Protection du fichier .env :
chmod 600 /opt/livestream/.env
chown deploy:deploy /opt/livestream/.env
echo '.env' >> /opt/livestream/.gitignore
```

**Chiffrement sops + age (recommande) :**

```bash
apt-get install -y age
wget -O /usr/local/bin/sops https://github.com/getsops/sops/releases/latest/download/sops-v3.9.0.linux.amd64
chmod +x /usr/local/bin/sops

age-keygen -o /opt/livestream/age-key.txt

sops --age $(cat /opt/livestream/age-key.txt | grep 'public key' | awk '{print $4}') \
  -e /opt/livestream/.env > /opt/livestream/.env.enc

# Dechiffrer :
SOPS_AGE_KEY_FILE=/opt/livestream/age-key.txt sops -d /opt/livestream/.env.enc > /opt/livestream/.env
```

> **Phase 2+ :** Remplacer `.env` par HashiCorp Vault en mode agent. Non necessaire pour Phase 1A.

### 6.6 Cloudflare WAF

```
Regle 1 -- Rate limit viewers (via Cloudflare Rules > Rate Limiting)
  IF : URI path contains "/live/"
  AND : Requests per 10 seconds > 30 per IP
  THEN : Block for 60 seconds
  REASON : Un viewer legitime ne fait qu'1 requete (le stream HTTP-FLV est continu)

Regle 2 -- Block non-browser user agents
  IF : URI path contains "/live/"
  AND : User Agent does NOT contain ("VLC" OR "MobileVLCKit" OR "libVLC" OR "Lavf" OR "stagefright")
  THEN : Challenge (CAPTCHA)

Regle 3 -- Geo-block pays non cibles (optionnel)
  IF : Country NOT IN (fr, us, gb, de, ng, ci, sn, cm, cd, ke, za, ...)
  THEN : Block
```

```nginx
# SRS Edge : rate limiting natif non supporte sur HTTP-FLV
# Protection assuree par le firewall whitelist (seuls CDN IPs accedent)
```

> **Double protection :** Le firewall whitelist empeche tout acces direct aux Edge. Le WAF Cloudflare protege les viewers qui passent par le CDN. Les deux ensemble = zero surface d'attaque.

### 6.7 IPv6

> **Pourquoi en 2026 :** En Afrique, l'espace IPv4 est sature. Beaucoup d'operateurs mobiles (MTN, Orange, Airtel) utilisent IPv6-only ou dual-stack.

```bash
#!/bin/bash
# /opt/ops/enable-ipv6.sh

ip -6 addr show scope global

# Si pas d'IPv6 visible, verifier :
cat /etc/netplan/01-netcfg.yaml
# Doit contenir :
#   addresses:
#     - 2a01:xxxx:xxxx::1/64
#   gateway6: fe80::1

echo 'net.ipv6.conf.all.forwarding=1' >> /etc/sysctl.conf
sysctl -p

ping6 -c3 google.com && echo "IPv6 OK"
```

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

> Cloudflare active IPv6 par defaut. Bunny CDN et Gcore CDN supportent IPv6 nativement. Rien a configurer cote CDN.

---

## 7. Monitoring et observabilite

### 7.1 Stack

```
Prometheus → scrape toutes les 15s
Grafana    → dashboards temps reel
AlertManager → PagerDuty + Slack
intel_gpu_top → metriques QSV par EX44
node_exporter → metriques systeme tous serveurs
Orchestrateur → /metrics endpoint Prometheus
Loki + Promtail → logs centralises
```

### 7.2 Alertes Prometheus

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
          summary: "Capacite QSV a {{ $value | humanizePercentage }} -- commander +5 EX44"

      - alert: SRSOriginDown
        expr: up{job="srs-origin"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "SRS Origin DOWN -- verifier OVH Rise-1"

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
          summary: "FFmpeg crash rate eleve sur {{ $labels.instance }}"
```

### 7.3 Verification QSV en temps reel

```bash
# Verifier la charge QSV sur chaque EX44
for ip in $(seq 1 20 | xargs -I{} echo "10.0.2.{}"); do
    streams=$(ssh root@$ip "ps aux | grep -c '[f]fmpeg'" 2>/dev/null || echo "0")
    echo "$ip : $streams streams actifs"
done

# Verifier la bande passante des Edge OVH
for ip in 10.0.3.1 10.0.3.2 10.0.3.3; do
    bw=$(ssh root@$ip "cat /sys/class/net/bond0/statistics/tx_bytes" 2>/dev/null)
    echo "Edge $ip : TX=$bw bytes"
done
```

### 7.4 Logs centralises -- Loki + Promtail

```yaml
# /etc/promtail/promtail-config.yaml -- Sur chaque EX44
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positons:
  filename: /tmp/positions.yaml

clients:
  - url: http://10.0.5.1:3100/loki/api/v1/push

scrape_configs:
  - job_name: ffmpeg
    static_configs:
      - targets: ['localhost']
        labels:
          job: ffmpeg-transcoder
          host: __HOSTNAME__
          env: production
          __path__: /var/log/ffmpeg/*.log

    pipeline_stages:
      - match:
          selector: '{job="ffmpeg-transcoder"}'
          stages:
            - drop:
                expression: '(frame=|fps=|bitrate=.*kbits|speed=)'
            - regex:
                expression: '(?P<level>WARNING|ERROR|FATAL|INFO)'
            - labels:
                level:
```

**Alertes Grafana sur logs FFmpeg :**

```yaml
# Broken pipe (connexion SRS Edge coupee)
expr: |
  count_over_time(
    {job="ffmpeg-transcoder"} |= "broken pipe" [5m]
  ) > 3
annotations:
  summary: "FFmpeg broken pipe sur {{ $labels.host }} -- verifier connexion SRS Edge"
```

```yaml
# Erreur QSV (drivers)
expr: |
  count_over_time(
    {job="ffmpeg-transcoder"} |= "QSV" |= "error" [5m]
  ) > 0
annotations:
  summary: "Erreur QSV sur {{ $labels.host }} -- verifier drivers intel-media-va-driver"
```

### 7.5 Latence End-to-End Viewer

**Metrique cote client (iOS) :**

```swift
// iOS -- MobileVLCKit : mesurer le Time-To-First-Frame (TTFF)
let startTime = CFAbsoluteTimeGetCurrent()
player.play()

func mediaPlayerStateChanged(_ notification: Notification) {
    if player.isPlaying {
        let ttff = CFAbsoluteTimeGetCurrent() - startTime
        Analytics.track("live_ttff", ["ttff_ms": Int(ttff * 1000), "quality": currentQuality])
    }
}
// Cible : TTFF < 1.5 secondes en 4G, < 800ms en WiFi
```

**Metrique cote serveur :**

```python
# Ajouter dans orchestrator.py -- endpoint /metrics
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
    summary: "Allocation FFmpeg trop lente ({{ $value }}s) -- cluster QSV surcharge ?"
```

**Dashboard Grafana -- Qualite Viewer :**

```promql
# Panels recommandes :
# 1. TTFF moyen (envoye par le client iOS/Android)
histogram_quantile(0.95, rate(live_ttff_seconds_bucket[5m]))

# 2. Latence allocation FFmpeg (serveur)
ffmpeg_allocation_latency_seconds

# 3. Nombre de viewers par qualite (ABR distribution)
sum by (quality) (live_viewers_by_quality)

# 4. Taux d'erreur viewer (connexions echouees / total)
rate(live_connection_errors_total[5m]) / rate(live_connection_total[5m])
```

| Metrique | Valeur cible |
|----------|-------------|
| Lives simultanes max | 500 |
| Streams actifs (80%) | 400 |
| EX44 Phase 1 | 20 (scaler si >300 streams actifs) |
| Latence live end-to-end | < 3s (HTTP-FLV) |
| Viewers max par live | Illimite (CDN scale auto) |
| Cold start lazy | < 5 secondes |
| Recovery FFmpeg crash | < 5s (Supervisor autorestart) |

---

## 8. Scaling intelligent

### 8.1 Architecture du systeme

```
Prometheus → AlertManager → Webhook → scaling-advisor.py → Slack (instructions exactes)
                                           ↓
                                    Optionnel : ordre auto via Hetzner Robot API
```

Le scaling-advisor recoit les metriques Prometheus et consulte le tableau de scaling encode en dur pour determiner EXACTEMENT le palier suivant.

### 8.2 Guide de scaling -- Que commander a chaque seuil

> **Principe :** Surveiller le dashboard Prometheus (`active_ffmpeg_streams_total`). Des qu'un seuil approche a 80%, commander les ressources du palier suivant. Chaque action est independante, ajouter UNIQUEMENT ce qui est liste.

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

**Formule de calcul rapide :**
```
Nombre EX44 necessaires = ceil(lives_actifs x 0.80 / 15)
Nombre Edge necessaires = ceil(EX44_total x 6.3 Mbps / 1 800 Mbps_par_Rise2)
   → avec 2Gbps Rise-2 des que tu depasses 300 lives

Ex : 500 lives x 80% = 400 actifs / 15 = 27 EX44 → arrondir a 34 (marge 25%)
```

**Quand upgrader Edge de 1Gbps a 2Gbps OVH :**
```
Seuil = quand chaque Rise-2 depasse 800 Mbps sortant vers CDN
Mesurer avec : iftop -i bond0 sur chaque Edge
Ou Grafana : node_network_transmit_bytes_total > 100 MB/s par noeud
Upgrade OVH : Control Panel → Serveur → Options reseau → Bande passante garantie
```

> **Regle d'or :** Commander le palier suivant quand on est a **70% du palier actuel** (pas 100%). Ex : si 3 EX44 pour 45 lives, commander les 2 suivants a **32 lives actifs** (70% de 45).

### 8.3 Le Scaling Advisor -- Le cerveau (Python)

```python
#!/usr/bin/env python3
# /opt/ops/scaling-advisor.py -- Micro-service qui encode le tableau de scaling
# Tourne sur le serveur Monitoring · Port 9099 · Recoit les webhooks AlertManager
import os, json, math, requests
from flask import Flask, request, jsonify

app = Flask(__name__)
SLACK_WEBHOOK = os.environ["SLACK_WEBHOOK_URL"]


# LE TABLEAU DE SCALING -- Encode directement depuis le document
# Chaque entree : (seuil_lives, ex44_total, edge_total, edge_bw, actions)

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
    return {
        "lives": current_lives,
        "ex44": math.ceil(current_lives * 0.80 / 15 * 1.25),
        "edge": math.ceil(current_lives / 150),
        "edge_bw": "2Gbps",
        "origin": 2,
        "phase": "beyond"
    }

def get_next_tier(current_lives: int) -> dict:
    """Trouve le prochain palier a atteindre."""
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
        actions.append("Upgrader TOUS les Edge de 1Gbps → 2Gbps (OVH Control Panel → Options reseau)")

    if origin_needed > 0:
        actions.append(f"Commander +{origin_needed} OVH Rise-1 Origin (HA cluster)")
        actions.append("   → Decommenter srs-origin-2 dans haproxy.cfg et redeployer")

    if not actions:
        actions.append("Infra suffisante pour ce palier -- rien a commander")

    return actions

@app.route("/webhook/scaling", methods=["POST"])
def scaling_webhook():
    """Recoit les alertes AlertManager et envoie des instructions Slack exactes."""
    data = request.json or {}
    
    for alert in data.get("alerts", []):
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        
        active_streams = float(annotations.get("active_streams", 0))
        ex44_count = int(annotations.get("ex44_count", 0))
        edge_count = int(annotations.get("edge_count", 0))
        origin_count = int(annotations.get("origin_count", 1))
        edge_bw = annotations.get("edge_bw", "1Gbps")
        
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
        
        trigger_pct = int(active_streams / max(ex44_count * 15, 1) * 100)
        
        slack_msg = {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text",
                    "text": f"SCALING ALERT — {estimated_lives} lives estimes"}},
                {"type": "section", "text": {"type": "mrkdwn",
                    "text": (
                        f"*Etat actuel :* {active_streams:.0f} streams actifs "
                        f"({trigger_pct}% de capacite)\n"
                        f"*Infra actuelle :* {ex44_count} EX44 · {edge_count} Edge "
                        f"({edge_bw}) · {origin_count} Origin\n"
                        f"*Palier actuel :* Phase {current_tier['phase']} "
                        f"(max {current_tier['lives']} lives)\n"
                        f"*Prochain palier :* Phase {next_tier['phase']} "
                        f"(max {next_tier['lives']} lives)"
                    )}},
                {"type": "divider"},
                {"type": "section", "text": {"type": "mrkdwn",
                    "text": "*Actions a faire MAINTENANT :*\n" +
                            "\n".join(f"• {a}" for a in actions)}},
                {"type": "context", "elements": [{"type": "mrkdwn",
                    "text": f"Rappel : EX44 = livre en quelques minutes (Allemagne) · "
                            f"Rise-2 OVH = livraison 24-48h · "
                            f"Hot Standby EX44 absorbe le pic en attendant"}]}
            ]
        }
        
        requests.post(SLACK_WEBHOOK, json=slack_msg, timeout=10)
    
    return jsonify(status="ok")

@app.route("/scaling/status", methods=["GET"])
def scaling_status():
    """Endpoint pour verifier l'etat du scaling advisor."""
    return jsonify(
        status="running",
        table_entries=len(SCALING_TABLE),
        max_capacity=SCALING_TABLE[-1]["lives"]
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9099)
```

### 8.4 Alertes Prometheus multi-niveaux

```yaml
# /etc/prometheus/alerts/scaling-intelligence.yml
# Ces alertes envoient les metriques EXACTES au scaling-advisor via webhook

groups:
  - name: scaling_intelligence
    rules:
      # --- Alerte principale : capacite QSV a 70% ---
      - alert: ScalingNeeded_QSV
        expr: |
          sum(active_ffmpeg_streams_total) /
          (count(up{job="qsv-workers"}) * 15) > 0.70
        for: 5m
        labels:
          severity: warning
          component: qsv
        annotations:
          summary: "Cluster QSV a {{ $value | humanizePercentage }}"
          active_streams: "{{ with query \"sum(active_ffmpeg_streams_total)\" }}{{ . | first | value }}{{ end }}"
          ex44_count: "{{ with query \"count(up{job='qsv-workers'})\" }}{{ . | first | value }}{{ end }}"
          edge_count: "{{ with query \"count(up{job='srs-edge'})\" }}{{ . | first | value }}{{ end }}"
          origin_count: "{{ with query \"count(up{job='srs-origin'})\" }}{{ . | first | value }}{{ end }}"

      # --- Alerte critique : Hot Standby en cours d'utilisation ---
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
          summary: "Le Hot Standby EX44 est maintenant actif -- commander immediatement !"

      # --- Alerte bande passante Edge ---
      - alert: EdgeBandwidthHigh
        expr: |
          max(rate(node_network_transmit_bytes_total{job="srs-edge",device="bond0"}[5m])) * 8
          > 800000000
        for: 10m
        labels:
          severity: warning
          component: edge_bandwidth
        annotations:
          summary: "SRS Edge {{ $labels.instance }} a {{ $value | humanize }}bps -- upgrader vers 2Gbps"

      # --- Alerte Origin : max_connections approche ---
      - alert: OriginConnectionsHigh
        expr: |
          srs_connections_total{job="srs-origin"} /
          srs_max_connections{job="srs-origin"} > 0.70
        for: 5m
        labels:
          severity: warning
          component: origin
        annotations:
          summary: "SRS Origin a {{ $value | humanizePercentage }} de connexions max -- ajouter un 2e Origin"
```

### 8.5 AlertManager -- Route vers le scaling-advisor

```yaml
# /etc/alertmanager/alertmanager.yml
route:
  group_by: ['alertname', 'component']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 1h
  receiver: 'slack-default'
  routes:
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
        title: 'HOT STANDBY ACTIVE -- COMMANDER MAINTENANT'
```

### 8.6 Script Hetzner Robot API -- Commande automatique

```bash
#!/bin/bash
# /opt/ops/order-ex44.sh -- Commande un ou plusieurs EX44 via Hetzner Robot API
# Doc API : https://robot.hetzner.com/doc/webservice/en.html
# Usage : ./order-ex44.sh [nombre]    (defaut : 1)
#
# EX44 standard en Allemagne = livre en quelques minutes (source : docs.hetzner.com mars 2026)
# Customise (hardware additionnel) = 2-3 jours ouvres. On commande TOUJOURS standard.
# Le add-ex44.sh sera lance automatiquement apres commande.

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
  echo "[$i/$COUNT] EX44 commande -- Order ID : $ORDER_ID"

  curl -s -X POST "$SLACK_WEBHOOK_URL" \
    -H 'Content-type: application/json' \
    --data "{\"text\": \"EX44 #$i/$COUNT commande (ID: $ORDER_ID). Livre en quelques minutes (Allemagne).\"}"
done
```

### 8.7 Overflow physique -- Le Hot Standby EX44

> **Pourquoi on ne fait PAS d'overflow sur Cloud VM (CCX33) :** Les instances cloud n'ont pas de puce graphique (Intel QSV). Elles transcoderaient via CPU (`libx264`) et s'effondreraient a 3-5 streams maximum avec une qualite instable. Inacceptable en production.

**La vraie solution : un EX44 de secours dedie.**
Pour seulement **49.00 EUR/mois**, on maintient en permanence 1 serveur EX44 de plus que necessaire, configure et connecte au cluster, mais garde vide (priorite basse dans l'orchestrateur Redis).

* **Capacite immediate :** 15-20 streams instantanes de marge en cas de pic inattendu.
* **Le temps tampon :** Ce serveur encaisse l'overflow instantanement pendant que le nouveau EX44 est commande et provisionne (~10-15 minutes total en Allemagne).
* **Alerte dediee :** Des qu'un seul stream est route sur le Hot Standby, l'alerte `HotStandbyActivated` (severity: critical) se declenche et le scaling-advisor envoie immediatement les instructions exactes sur Slack.

### 8.8 Scenario reel concret

```
Situation : 3 EX44 (Phase 1A, 45 lives max). 32 lives actifs.

1. Prometheus mesure : active_ffmpeg_streams_total = 26 (32 x 80% lazy)
2. Calcul : 26 / (3 x 15) = 57.8% → pas encore 70% → rien

Situation : 42 lives actifs

1. Prometheus mesure : active_ffmpeg_streams_total = 34
2. Calcul : 34 / (3 x 15) = 75.5% → > 70% pendant 5 min → ALERTE !
3. AlertManager envoie au scaling-advisor (port 9099)
4. Le scaling-advisor consulte le tableau :
   - Lives estimes : 34 / 0.80 = 42
   - Palier actuel : 45 (Phase 1A, 3 EX44)
   - Palier suivant : 75 (Phase 1A+, 5 EX44)
5. Actions calculees : "+2 EX44"
6. Message Slack envoye :

   SCALING ALERT — 42 lives estimes
   ─────────────────────────────────
   Etat actuel : 34 streams actifs (75% de capacite)
   Infra actuelle : 3 EX44 · 1 Edge (1Gbps) · 1 Origin
   Palier actuel : Phase 1A (max 45 lives)
   Prochain palier : Phase 1A+ (max 75 lives)
   ─────────────────────────────────
   Actions a faire MAINTENANT :
   • Commander +2 EX44 (total : 5)
   •    → Lancer : for i in $(seq 1 2); do /opt/ops/order-ex44.sh; done
   ─────────────────────────────────
   Hot Standby absorbe le pic en attendant (15 streams de marge)
```

---

## 9. Budgets par phase

### 9.1 Comparaison architectures (Phase 1)

| Poste | Avant (ancienne archi) | EX44 QSV + Rise-2 (actuel) | Gain/mois |
|-------|----------------------|----------------------------|-----------| 
| Origin | ~95 EUR | **Rise-1 = 65 EUR** | -30 EUR |
| Edge (x3) | ~510 EUR | **Rise-2 2Gbps = ~310 EUR** | **-200 EUR** |
| FFmpeg (x20) | ~4 246 EUR | **EX44 QSV = 946 EUR** | **-3 300 EUR** |
| **TOTAL infra** | **~4 851 EUR** | **~1 321 EUR** | **-3 530 EUR/mois** |

> **Economie annuelle : -42 360 EUR/an. Sur 3 ans : -127 080 EUR.**

### 9.2 Leviers d'economie additionnels

| Action | Economie | Effort |
|--------|---------|--------|
| Lazy transcode (integre) | -40 a -60% charge EX44 | Deja fait |
| Bunny Volume Network (>100TB) | $0.005/GB vs $0.010 = **-50%** | Auto |
| Gcore PRO (100 EUR/mois) | 5TB inclus, -33% vs FREE | Email |
| Hetzner Server Auction | -5 a -10% EX44 | Verifier Robot |
| P2P CDN (Streamroot/Peer5) | -50 a -70% CDN | Phase 3 |

### 9.3 Phase 1A -- Demarrage reel : 45 lives simultanes

> **Point de depart.** Infrastructure minimale mais 100% compatible avec le reste de l'archi. Tout le code, les configs, les scripts sont identiques -- on ajoute juste des serveurs au fur et a mesure.

**Calcul :**
```
45 lives actifs x lazy transcoding 80% = 36 streams encodes simultanement
36 streams / 15 par EX44 = 3 EX44 suffisent (marge : 45 - 36 = 9 slots libres)
Bande passante Edge : 36 streams x 6.3 Mbps = 227 Mbps → 1 Rise-2 1 Gbps suffit
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

### 9.4 Phase 2 -- Scale : 500 a 1 000 lives

**Sizing exact :**
```
Streams actifs = 1 000 x 80% = 800
EX44 = 800 / 15 = 53 + 5 buffer = 58 EX44
```

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

### 9.5 Phase 3 -- Production : 1 000 a 2 000 lives

**Sizing exact :**
```
Conservateur (80%) : 2 000 x 80% = 1 600 streams → 107 EX44 + 10 buffer = 117
Optimiste (60%)    : 2 000 x 60% = 1 200 streams →  80 EX44 +  8 buffer =  88

Recommandation : demarrer a 88, scaler selon metriques reelles
```

| Role | Modele | Qte | Total/mois |
|------|--------|-----|------------|
| SRS Origin | OVH Rise-1 | **2** (HA) | 96.88 EUR |
| FFmpeg QSV | Hetzner EX44 | **88-117** | 4 312-5 733 EUR |
| SRS Edge | OVH Rise-2 2Gbps | **4** | ~620 EUR |
| Monitoring (HA) | AX41-NVMe | 2 | 88.00 EUR |
| Redis HA | CPX32 | 2 | 28.98 EUR |
| **Sous-total serveurs** | | | **~5 146-6 567 EUR** |
| CDN total | Bunny + Gcore | -- | 800-5 000 EUR |
| **TOTAL Phase 3** | | | **~5 946-11 567 EUR** |

### 9.6 Recapitulatif scalabilite -- Les 3 phases

| Phase | Lives max | Streams actifs 80% | EX44 | Total/mois (serveurs+CDN) |
|-------|----------|---------------------|------|---------------------------|
| **Phase 1A** | 45 | 36 | **3+1** | ~370-560 EUR |
| **Phase 1 MVP** | 300 | 240 | **20+1** | ~1 622-2 722 EUR |
| **Phase 2** | 1 000 | 800 | **50+** | ~3 385-5 085 EUR |
| **Phase 3** | 2 000 | 1 600 | **88-117** | ~5 946-11 567 EUR |

### 9.7 Budget consolide -- Toutes les phases


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

### 9.8 Cost Monitoring et Budget Alerts

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
          summary: "Cluster QSV a moins de 20% d'utilisation depuis 1h -- retirer des EX44 ?"

      # Alert bande passante Edge anormalement haute
      - alert: EdgeBandwidthAnomaly
        expr: |
          node_network_transmit_bytes_total{device="bond0"} / 125000000 > 1500
        for: 10m
        annotations:
          summary: "SRS Edge {{ $labels.instance }} a {{ $value }} Mbps -- pic de trafic ou attaque ?"
```

**Dashboard Grafana -- Indicateurs financiers en temps reel :**
```promql
# Cout Bunny estime en temps reel (total bytes sortant Edge x prix)
(sum(increase(node_network_transmit_bytes_total{job="srs-edge"}[1h])) / 1e9) * 0.005
# → Afficher en panel "EUR/heure CDN estime"

# Nombre d'EX44 actifs vs necessaires
count(up{job="qsv-workers"}) - ceil(sum(active_ffmpeg_streams_total) / 15 * 1.25)
# → Positif = surplus (gaspillage), negatif = manque (urgence commande)
```


## 10. Operations

### 10.1 CI/CD Pipeline -- GitHub Actions

```yaml
# .github/workflows/deploy-infra.yml
name: Deploy Live Infrastructure

on:
  push:
    branches: [ main ]
  workflow_dispatch:

env:
  ROLLBACK_TIMEOUT: 30

jobs:
  # ── Etape 1 : Validation (lint) ──
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

  # ── Etape 2 : Deploiement HAProxy + Keepalived ──
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
            cp haproxy.cfg haproxy.cfg.bak
            git pull
            if ! haproxy -c -f /etc/haproxy/haproxy.cfg; then
              echo "Config HAProxy invalide -- rollback"
              cp haproxy.cfg.bak haproxy.cfg
              exit 1
            fi
            haproxy -f /etc/haproxy/haproxy.cfg -p /var/run/haproxy.pid -sf $(cat /var/run/haproxy.pid)
            sleep 2
            if ! echo 'show stat' | socat stdio /var/run/haproxy.sock > /dev/null 2>&1; then
              echo "HAProxy ne repond pas -- rollback"
              cp haproxy.cfg.bak haproxy.cfg
              haproxy -f /etc/haproxy/haproxy.cfg -p /var/run/haproxy.pid -sf $(cat /var/run/haproxy.pid)
              exit 1
            fi
            echo "HAProxy deploy OK"

  # ── Etape 3 : Deploiement SRS Origin + Edge ──
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
            if ! ./objs/srs -t -c /etc/srs/srs.conf; then
              echo "Config SRS invalide -- rollback"
              cp srs.conf.bak srs.conf
              exit 1
            fi
            systemctl restart srs
            sleep 3
            if ! curl -sf http://localhost:1985/api/v1/versions > /dev/null; then
              echo "SRS ne repond pas -- rollback"
              cp srs.conf.bak srs.conf
              systemctl restart srs
              exit 1
            fi
            echo "SRS Origin deploy OK"

      - name: Deploy SRS Edge (avec rollback)
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.EDGE_IPS }}
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

  # ── Etape 4 : Deploiement FFmpeg Workers ──
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
            bash -n transcode-qsv.sh || {
              cp transcode-qsv.sh.bak transcode-qsv.sh
              exit 1
            }
            supervisorctl reread && supervisorctl update
            echo "FFmpeg scripts deploy OK (streams actifs non coupes)"

  # ── Etape 5 : Deploiement Orchestrateur ──
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
            docker build -t orchestrator:$(git rev-parse --short HEAD) .
            docker tag orchestrator:$(git rev-parse --short HEAD) orchestrator:latest
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
              echo "Orchestrator ne repond pas -- rollback"
              docker stop orchestrator && docker rm orchestrator
              docker rename orchestrator-old orchestrator
              docker start orchestrator
              exit 1
            fi
            docker rm orchestrator-old 2>/dev/null || true
            echo "Orchestrator deploy OK"
```

> **GitHub Secrets a configurer :**
> `LB1_IP`, `LB2_IP`, `ORIGIN_IP`, `EDGE_IPS`, `WORKER_IPS`, `DEPLOY_SSH_KEY` (cle ED25519 dediee deploy)
> Ne jamais utiliser la cle root -- creer un user `deploy` avec les droits necessaires sur chaque serveur.

### 10.2 Backup et Disaster Recovery

**Strategie :**

| Donnee | Frequence | Destination | RTO | RPO |
|--------|----------|-------------|-----|-----|
| Config SRS (srs.conf) | Git commit | GitHub | 0 (git pull) | 0 |
| Config HAProxy | Git commit | GitHub | 0 | 0 |
| Config FFmpeg scripts | Git commit | GitHub | 0 | 0 |
| Redis dump (pool workers) | 1h | Hetzner Object Storage | 5 min | 1h |
| Orchestrateur (Flask + .env) | Quotidien | Hetzner Object Storage | 30 min | 24h |

> **Principe :** Toutes les configs sont dans Git. Un nouveau serveur = `git pull` + deploy. Aucune config manuelle sur les serveurs.

**Hetzner Object Storage (S3_BUCKET) :**

Hetzner Object Storage est un stockage objet compatible avec l'API S3 d'Amazon. La variable `S3_BUCKET` dans le script pointe vers un bucket a creer dans la console Hetzner (Cloud Console > Object Storage > Create Bucket). L'outil `aws s3` fonctionne avec n'importe quel provider compatible S3 via le flag `--endpoint-url`. Le endpoint `fsn1.your-objectstorage.com` est celui de Hetzner pour le datacenter de Falkenstein (FSN1).

```bash
#!/bin/bash
# /opt/ops/backup-daily.sh -- execute via cron 0 3 * * *
set -euo pipefail

DATE=$(date +%Y%m%d-%H%M)
BACKUP_DIR="/tmp/backup-$DATE"
S3_BUCKET="s3://shopfeed-backup/live-infra"

mkdir -p "$BACKUP_DIR"

# Redis : snapshot RDB
redis-cli -h 10.0.4.1 BGSAVE
sleep 2
cp /var/lib/redis/dump.rdb "$BACKUP_DIR/redis-$DATE.rdb"

# Orchestrateur : copier .env
cp /opt/livestream/.env "$BACKUP_DIR/env-$DATE.bak"

# Upload vers Hetzner Object Storage (compatible S3)
aws s3 cp "$BACKUP_DIR/" "$S3_BUCKET/$DATE/" --recursive \
    --endpoint-url https://fsn1.your-objectstorage.com

# Garder 30 jours de backups seulement
aws s3 ls "$S3_BUCKET/" | sort | head -n -30 | \
    awk '{print $4}' | xargs -I{} aws s3 rm "$S3_BUCKET/{}"

echo "Backup $DATE OK"
```

**Disaster Recovery -- EX44 crash total :**
```bash
# 1. Commander un nouveau EX44 chez Hetzner (livre en quelques minutes en Allemagne)
# 2. Des livraison : lancer le script de provisioning
/opt/ops/add-ex44.sh <NOUVELLE_IP>
# Ce script installe tout (QSV drivers, FFmpeg, Supervisor) + enregistre dans Redis
# 3. Aucun stream ne se perd : le lazy transcoding relancera FFmpeg au prochain viewer
echo "Recovery complet en < 10 minutes (commande + provisioning)"
```

**Perte totale d'un datacenter :**

| Composant perdu | Impact | Recovery |
|----------------|--------|----------|
| **Tous les EX44 (Hetzner)** | 0 transcodage = pas de qualite ABR | 1. Commander 3+ EX44 dans un autre DC (NBG1, HEL1). 2. `add-ex44.sh` sur chacun. 3. L'orchestrateur redirige automatiquement. |
| **Rise-1 Origin (OVH)** | Aucun nouveau stream accepte | 1. Commander Rise-1 dans un autre DC OVH. 2. `git pull` + installer SRS + orchestrateur. 3. DNS Cloudflare : mettre a jour l'IP Origin. |
| **Rise-2 Edge (OVH)** | CDN ne peut plus pull | 1. Commander Rise-2 dans un autre DC. 2. Installer SRS Edge. 3. Mettre a jour les origins Bunny/Gcore. |
| **Redis (Hetzner Cloud)** | Perte etat workers. Pas de nouvelles allocations. | 1. Redeployer CPX11. 2. Restaurer le dump RDB depuis Object Storage. 3. Relancer l'orchestrateur. |

```bash
#!/bin/bash
# /opt/ops/dr-rebuild-from-scratch.sh
set -euo pipefail

echo "1. Cloner le depot de configs..."
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

echo "Reconstruction complete."
```

### 10.3 Ajouter un EX44 au cluster

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

### 10.4 Checklist deploiement initial (Phase 1A -- 45 lives)

```
[ ] OVH Rise-1 commande + HAProxy installe (:1935 + :10080)
[ ] SRS Origin installe sur Rise-1 (:11935) + hooks vers orchestrateur :8085
[ ] Orchestrateur Flask installe + Redis CPX11 deploye (10.0.4.1)
[ ] OVH Rise-2 x1 commande (1 Gbps standard suffit pour Phase 1A)
[ ] vRack OVH configure entre Rise-1 et Rise-2 (trafic prive gratuit)
[ ] Firewall Edge whitelist : seules IPs Gcore + Bunny + Hetzner workers autorisees
[ ] BBR TCP active sur tous les serveurs OVH (sysctl -p)
[ ] Hetzner EX44 x3 commandes (3 x 15 = 45 streams max)
[ ] Sur chaque EX44 : apt install intel-media-va-driver-non-free libmfx1 vainfo libva-drm2
[ ] usermod -aG render <ffmpeg-user> sur chaque EX44
[ ] vainfo | grep H264 → OK sur chaque EX44
[ ] ffmpeg -filters | grep vpp_qsv → OK sur chaque EX44
[ ] JWT_SECRET configure en variable d'environnement
[ ] Bunny CDN : 1 origin pointe sur Rise-2 Edge
[ ] Gcore CDN : Pull Zone configuree pointant vers Edge IP
[ ] Cloudflare Geo DNS : Afrique/Asie → Gcore · EU/USA → Bunny
[ ] Prometheus leger : active_ffmpeg_streams_total visible
[ ] Test E2E : HaishinKit → RTMP → SRS → FFmpeg QSV → CDN → viewer OK
[ ] Test Lazy : verifier que FFmpeg ne demarre que quand viewer arrive
[ ] Commander EX44 #4 et #5 des que 32 lives actifs simultanes sur Grafana
```

> Phase 1A → next step : Quand Grafana montre `active_ffmpeg_streams_total > 30`, lancer immediatement `add-ex44.sh` pour les serveurs #4 et #5. C'est transparent -- le cluster s'agrandit a chaud sans coupure.

---

## 11. Annexes

### 11.1 Variables d'environnement critiques

| Variable | Description | Ou la configurer |
|----------|-------------|-----------------|
| `JWT_SECRET` | Secret de signature JWT pour authentification RTMP/FLV | `/opt/livestream/.env` |
| `SUPERVISOR_USER` | Username Supervisor XML-RPC | `/opt/livestream/.env` |
| `SUPERVISOR_PASS` | Password Supervisor XML-RPC | `/opt/livestream/.env` |
| `REDIS_URL` | URL de connexion Redis | `/opt/livestream/.env` |
| `HETZNER_ROBOT_USER` | Login Hetzner Robot API | `/opt/livestream/.env` |
| `HETZNER_ROBOT_PASS` | Password Hetzner Robot API | `/opt/livestream/.env` |
| `SLACK_WEBHOOK_URL` | Webhook Slack pour alertes scaling | `/opt/livestream/.env` |
| `S3_BUCKET` | Bucket Hetzner Object Storage pour backups | Script `backup-daily.sh` |

### 11.2 Recapitulatif des scripts operationnels

| Script | Chemin | Usage |
|--------|--------|-------|
| `transcode-qsv.sh` | `/opt/livestream/transcode-qsv.sh` | Commande FFmpeg QSV 4 profils ABR |
| `add-ex44.sh` | `/opt/ops/add-ex44.sh` | Provisioning automatique d'un nouveau EX44 |
| `order-ex44.sh` | `/opt/ops/order-ex44.sh` | Commande EX44 via Hetzner Robot API |
| `backup-daily.sh` | `/opt/ops/backup-daily.sh` | Backup Redis + .env vers Object Storage |
| `dr-rebuild-from-scratch.sh` | `/opt/ops/dr-rebuild-from-scratch.sh` | Reconstruction totale apres perte DC |
| `harden-ssh.sh` | `/opt/ops/harden-ssh.sh` | Hardening SSH post-provisioning |
| `kernel-tuning.sh` | `/opt/ops/kernel-tuning.sh` | Optimisation TCP BBR + buffers reseau |
| `enable-ipv6.sh` | `/opt/ops/enable-ipv6.sh` | Activation IPv6 sur tous les serveurs |
| `scaling-advisor.py` | `/opt/ops/scaling-advisor.py` | Micro-service scaling intelligent (port 9099) |
| `orchestrator.py` | `/opt/livestream/orchestrator.py` | Orchestrateur principal Flask (port 8085) |

### 11.3 Recapitulatif des fichiers de configuration

| Fichier | Chemin | Serveur |
|---------|--------|---------|
| HAProxy config | `/etc/haproxy/haproxy.cfg` | LB1, LB2 (CX33) |
| Keepalived config | `/etc/keepalived/keepalived.conf` | LB1, LB2 (CX33) |
| SRS Origin config | `/etc/srs/srs.conf` | OVH Rise-1 |
| SRS Edge config | `/etc/srs/srs-edge.conf` | OVH Rise-2 (x3) |
| Supervisor base config | `/etc/supervisor/conf.d/base.conf` | EX44 (chaque) |
| Promtail config | `/etc/promtail/promtail-config.yaml` | EX44 (chaque) |
| Prometheus alerts | `/etc/prometheus/alerts/*.yml` | Monitoring (AX41) |
| AlertManager config | `/etc/alertmanager/alertmanager.yml` | Monitoring (AX41) |
| Docker Orchestrateur | `/opt/livestream/Dockerfile` | OVH Rise-1 |
| Fichier .env | `/opt/livestream/.env` | OVH Rise-1 |
| GitHub Actions | `.github/workflows/deploy-infra.yml` | GitHub |

### 11.4 Ports reseau

| Port | Protocole | Service | Serveur |
|------|----------|---------|---------|
| 1935 | TCP | RTMP ingest (HAProxy) | LB (VIP) |
| 10080 | TCP | SRT ingest (HAProxy) | LB (VIP) |
| 11935 | TCP | SRS Origin RTMP interne | Rise-1 |
| 8080 | TCP | SRS HTTP-FLV | Rise-1, Rise-2 |
| 1985 | TCP | SRS HTTP API | Rise-1, Rise-2 |
| 8085 | TCP | Orchestrateur Flask | Rise-1 |
| 6379 | TCP | Redis | CPX11 (10.0.4.1) |
| 9001 | TCP | Supervisor XML-RPC | EX44 (chaque) |
| 9090 | TCP | Prometheus | Monitoring |
| 3000 | TCP | Grafana | Monitoring |
| 9093 | TCP | AlertManager | Monitoring |
| 3100 | TCP | Loki | Monitoring |
| 9099 | TCP | Scaling Advisor | Monitoring |
| 8404 | TCP | HAProxy Stats | LB1, LB2 |

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
