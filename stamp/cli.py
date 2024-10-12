from omegaconf import OmegaConf, DictConfig
import argparse
from pathlib import Path
import os
from typing import Iterable, Optional
import shutil
import torch
import timm

NORMALIZATION_TEMPLATE_URL = "https://github.com/Avic3nna/STAMP/blob/main/resources/normalization_template.jpg?raw=true"
CTRANSPATH_WEIGHTS_URL = "https://drive.google.com/u/0/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX&export=download"
CHIEF_WEIGHTS_URL = "https://drive.google.com/uc?id=1_vgRF1QXa8sPCOpJ1S9BihwZhXQMOVJc&export=download"
DEFAULT_RESOURCES_DIR = Path(__file__).with_name("resources")
DEFAULT_CONFIG_FILE = Path("config.yaml")
STAMP_FACTORY_SETTINGS = Path(__file__).with_name("config.yaml")

class ConfigurationError(Exception):
    pass

def _config_has_key(cfg: DictConfig, key: str):
    try:
        for k in key.split("."):
            cfg = cfg[k]
        if cfg is None:
            return False
    except KeyError:
        return False
    return True

def require_configs(cfg: DictConfig, keys: Iterable[str], prefix: Optional[str] = None):
    prefix = f"{prefix}." if prefix else ""
    keys = [f"{prefix}{k}" for k in keys]
    missing = [k for k in keys if not _config_has_key(cfg, k)]
    if len(missing) > 0:
        raise ConfigurationError(f"Missing required configuration keys: {missing}")

def create_config_file(config_file: Optional[Path]):
    """Create a new config file at the specified path (by copying the default config file)."""
    config_file = config_file or DEFAULT_CONFIG_FILE
    # Locate original config file
    if not STAMP_FACTORY_SETTINGS.exists():
        raise ConfigurationError(f"Default STAMP config file not found at {STAMP_FACTORY_SETTINGS}")
    # Copy original config file
    shutil.copy(STAMP_FACTORY_SETTINGS, config_file)
    print(f"Created new config file at {config_file.absolute()}")

def resolve_config_file_path(config_file: Optional[Path]) -> Path:
    """Resolve the path to the config file, falling back to the default config file if not specified."""
    if config_file is None:
        if DEFAULT_CONFIG_FILE.exists():
            config_file = DEFAULT_CONFIG_FILE
        else:
            config_file = STAMP_FACTORY_SETTINGS
            print(f"Falling back to default STAMP config file because {DEFAULT_CONFIG_FILE.absolute()} does not exist")
            if not config_file.exists():
                raise ConfigurationError(f"Default STAMP config file not found at {config_file}")
    if not config_file.exists():
        raise ConfigurationError(f"Config file {Path(config_file).absolute()} not found (run `stamp init` to create the config file or use the `--config` flag to specify a different config file)")
    return config_file

def run_cli(args: argparse.Namespace):
    # Handle init command
    if args.command == "init":
        create_config_file(args.config)
        return

    # Load YAML configuration
    config_file = resolve_config_file_path(args.config)
    cfg = OmegaConf.load(config_file)

    # Set environment variables
    if "STAMP_RESOURCES_DIR" not in os.environ:
        os.environ["STAMP_RESOURCES_DIR"] = str(DEFAULT_RESOURCES_DIR)
    
    match args.command:
        case "init":
            return # this is handled above
        case "setup":
            # Download normalization template
            normalization_template_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/normalization_template.jpg")
            normalization_template_path.parent.mkdir(parents=True, exist_ok=True)
            if normalization_template_path.exists():
                print(f"Skipping download, normalization template already exists at {normalization_template_path}")
            else:
                print(f"Downloading normalization template to {normalization_template_path}")
                import requests
                r = requests.get(NORMALIZATION_TEMPLATE_URL)
                with normalization_template_path.open("wb") as f:
                    f.write(r.content)

            # Download feature extractor model
            feat_extractor = cfg.preprocessing.feat_extractor
            if feat_extractor == 'ctp':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/ctranspath.pth")
            elif feat_extractor == 'chief-ctp':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/chief-ctp.pth")
            elif feat_extractor == 'uni':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/uni/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin")
            elif feat_extractor == 'provgp':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/prov-gigapath/pytorch_model.bin")
            elif feat_extractor == 'hibou-b':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/hibou-b/pytorch_model.bin")
            elif feat_extractor == 'hibou-l':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/hibou-l/pytorch_model.bin")
            elif feat_extractor == 'kaiko':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/kaiko-vitl14/pytorch_model.bin")
            elif feat_extractor == 'conch':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/conch/pytorch_model.bin")
            elif feat_extractor == 'phikon':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/phikon/pytorch_model.bin")
            elif feat_extractor == 'virchow':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/virchow/pytorch_model.bin")
            elif feat_extractor == 'virchow2':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/virchow2/pytorch_model.bin")
            elif feat_extractor == 'hoptimus0':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/hoptimus0/pytorch_model.bin")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if model_path.exists():
                print(f"Skipping download, feature extractor model already exists at {model_path}")
            else:
                if feat_extractor == 'ctp':
                    print(f"Downloading CTransPath weights to {model_path}")
                    import gdown
                    gdown.download(CTRANSPATH_WEIGHTS_URL, str(model_path))
                elif feat_extractor == 'chief-ctp':
                    print(f"Downloading CHIEF weights to {model_path}")
                    import gdown
                    gdown.download(CHIEF_WEIGHTS_URL, str(model_path))
                elif feat_extractor == 'uni':
                    print(f"Downloading UNI weights")
                    from uni.get_encoder import get_encoder
                    get_encoder(enc_name='uni', checkpoint='pytorch_model.bin', assets_dir=f"{os.environ['STAMP_RESOURCES_DIR']}/uni")
                elif feat_extractor == 'provgp':
                    print("Downloading ProvGP weights")
                    assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
                    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
                                        
                    model_name = 'prov-gigapath'
                    checkpoint = 'pytorch_model.bin'

                    ckpt_dir = os.path.join(assets_dir, model_name)
                    ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

                    # Ensure the directory exists
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Save the model
                    torch.save(model.state_dict(), ckpt_path)
                elif feat_extractor == 'hibou-b':
                    print(f"Downloading Hibou weights")
                    from transformers import AutoImageProcessor, AutoModel

                    # Define paths
                    assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
                    model_name = 'hibou-b'
                    checkpoint = 'pytorch_model.bin'

                    ckpt_dir = os.path.join(assets_dir, model_name)
                    ckpt_path = os.path.join(ckpt_dir, checkpoint)

                    # Ensure the directory exists
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Load model and processor
                    processor = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
                    hf_model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)

                    # Save the model state dict
                    torch.save(hf_model.state_dict(), ckpt_path)

                    # Save the processor
                    processor_path = os.path.join(ckpt_dir, 'processor')
                    processor.save_pretrained(processor_path)
                elif feat_extractor == 'hibou-l':
                    print(f"Downloading Hibou-L weights")
                    from transformers import AutoImageProcessor, AutoModel

                    # Define paths
                    assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
                    model_name = 'hibou-l'
                    checkpoint = 'pytorch_model.bin'

                    ckpt_dir = os.path.join(assets_dir, model_name)
                    ckpt_path = os.path.join(ckpt_dir, checkpoint)

                    # Ensure the directory exists
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Load model and processor
                    processor = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
                    hf_model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)

                    # Save the model state dict
                    torch.save(hf_model.state_dict(), ckpt_path)

                    # Save the processor
                    processor_path = os.path.join(ckpt_dir, 'processor')
                    processor.save_pretrained(processor_path)
                elif feat_extractor == 'kaiko':
                    print("Downloading Kaiko weights")
                    assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"

                    model_name = 'kaiko-vitl14'
                    checkpoint = 'pytorch_model.bin'

                    ckpt_dir = os.path.join(assets_dir, model_name)
                    ckpt_path = os.path.join(ckpt_dir, checkpoint)

                    # Ensure the directory exists
                    os.makedirs(ckpt_dir, exist_ok=True)
                    vitl14 = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitl14", trust_repo=True)
                    
                    # Save the model state dict
                    torch.save(vitl14.state_dict(), ckpt_path)

                elif feat_extractor == 'conch':
                    print("Downloading CONCH weights")
                    assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
                    from conch.open_clip_custom import create_model_from_pretrained
                    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")

                    model_name = 'conch'
                    checkpoint = 'pytorch_model.bin'

                    ckpt_dir = os.path.join(assets_dir, model_name)
                    ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

                    # Ensure the directory exists
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Save the model state dict
                    torch.save(model.state_dict(), ckpt_path)
                elif feat_extractor == 'phikon':
                    print("Downloading Phikon weights")
                    from transformers import AutoImageProcessor, AutoModel

                    # Define paths
                    assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
                    model_name = 'phikon'
                    checkpoint = 'pytorch_model.bin'

                    ckpt_dir = os.path.join(assets_dir, model_name)
                    ckpt_path = os.path.join(ckpt_dir, checkpoint)

                    # Ensure the directory exists
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Load model and processor
                    processor = AutoImageProcessor.from_pretrained("owkin/phikon")
                    model = AutoModel.from_pretrained("owkin/phikon")

                    # Save the model state dict
                    torch.save(model.state_dict(), ckpt_path)

                    # Save the processor
                    processor_path = os.path.join(ckpt_dir, 'processor')
                    processor.save_pretrained(processor_path)
                elif feat_extractor == 'virchow':
                    print("Downloading Virchow weights")
                    assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"

                    from timm.layers import SwiGLUPacked
                    # from huggingface_hub import login
                    # login()

                    model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
                                        
                    model_name = 'virchow'
                    checkpoint = 'pytorch_model.bin'

                    ckpt_dir = os.path.join(assets_dir, model_name)
                    ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

                    # Ensure the directory exists
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Save the model
                    torch.save(model.state_dict(), ckpt_path)
                elif feat_extractor == 'virchow2':
                    print("Downloading Virchow2 weights")
                    assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"

                    from timm.layers import SwiGLUPacked
                    # from huggingface_hub import login
                    # login()

                    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
                                        
                    model_name = 'virchow2'
                    checkpoint = 'pytorch_model.bin'

                    ckpt_dir = os.path.join(assets_dir, model_name)
                    ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

                    # Ensure the directory exists
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Save the model
                    torch.save(model.state_dict(), ckpt_path)
                elif feat_extractor == 'hoptimus0':
                    print("Downloading H-optimus-0 weights")
                    assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"

                    model = model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)          
                    model_name = 'hoptimus0'
                    checkpoint = 'pytorch_model.bin'

                    ckpt_dir = os.path.join(assets_dir, model_name)
                    ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

                    # Ensure the directory exists
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Save the model
                    torch.save(model.state_dict(), ckpt_path)
                
        case "config":
            print(OmegaConf.to_yaml(cfg, resolve=True))
        case "preprocess":
            require_configs(
                cfg,
                ["output_dir", "wsi_dir", "cache_dir", "microns", "cores", "norm", "del_slide", "only_feature_extraction", "device", "feat_extractor"],
                prefix="preprocessing"
            )
            c = cfg.preprocessing
            # Some checks
            normalization_template_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/normalization_template.jpg")
            if c.norm and not Path(normalization_template_path).exists():
                raise ConfigurationError(f"Normalization template {normalization_template_path} does not exist, please run `stamp setup` to download it.")
            if c.feat_extractor == 'ctp':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/ctranspath.pth"
            elif c.feat_extractor == 'chief-ctp':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/chief-ctp.pth"
            elif c.feat_extractor == 'uni':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/uni/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
            elif c.feat_extractor == 'provgp':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/prov-gigapath/pytorch_model.bin"
            elif c.feat_extractor == 'hibou-b':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/hibou-b/pytorch_model.bin"
            elif c.feat_extractor == 'hibou-l':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/hibou-l/pytorch_model.bin"
            elif c.feat_extractor == 'kaiko':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/kaiko-vitl14/pytorch_model.bin"
            elif c.feat_extractor == 'conch':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/conch/pytorch_model.bin"
            elif c.feat_extractor == 'phikon':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/phikon/pytorch_model.bin"
            elif c.feat_extractor == 'virchow':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/virchow/pytorch_model.bin"
            elif c.feat_extractor == 'virchow2':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/virchow2/pytorch_model.bin"
            elif c.feat_extractor == 'hoptimus0':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/hoptimus0/pytorch_model.bin"
          
            if not Path(model_path).exists():
                raise ConfigurationError(f"Feature extractor model {model_path} does not exist, please run `stamp setup` to download it.")

            from .preprocessing.wsi_norm import preprocess
            preprocess(
                output_dir=Path(c.output_dir),
                wsi_dir=Path(c.wsi_dir),
                model_path=Path(model_path),
                cache_dir=Path(c.cache_dir),
                feat_extractor=c.feat_extractor,
                # patch_size=c.patch_size,
                target_microns=c.microns,
                cores=c.cores,
                norm=c.norm,
                del_slide=c.del_slide,
                cache=c.cache if 'cache' in c else True,
                only_feature_extraction=c.only_feature_extraction,
                keep_dir_structure=c.keep_dir_structure if 'keep_dir_structure' in c else False,
                device=c.device,
                normalization_template=normalization_template_path
            )
        case "train":
            require_configs(
                cfg,
                ["output_dir", "feature_dir", "target_label", "cat_labels", "cont_labels"],
                prefix="modeling"
            )
            c = cfg.modeling
            from .modeling.marugoto.transformer.helpers import train_categorical_model_
            train_categorical_model_(clini_table=Path(c.clini_table), 
                                     slide_table=Path(c.slide_table),
                                     feature_dir=Path(c.feature_dir), 
                                     output_path=Path(c.output_dir),
                                     target_label=c.target_label, 
                                     cat_labels=c.cat_labels,
                                     cont_labels=c.cont_labels, 
                                     categories=c.categories)
        case "crossval":
            require_configs(
                cfg,
                ["output_dir", "feature_dir", "target_label", "cat_labels", "cont_labels", "n_splits"], # this one requires the n_splits key!
                prefix="modeling"
            )
            c = cfg.modeling
            from .modeling.marugoto.transformer.helpers import categorical_crossval_
            categorical_crossval_(clini_table=Path(c.clini_table), 
                                  slide_table=Path(c.slide_table),
                                  feature_dir=Path(c.feature_dir),
                                  output_path=Path(c.output_dir),
                                  target_label=c.target_label,
                                  cat_labels=c.cat_labels,
                                  cont_labels=c.cont_labels,
                                  categories=c.categories,
                                  n_splits=c.n_splits,
                                  model_arch=c.model_arch,
                                  pretrained=c.pretrained,
                                  freeze_base=c.freeze_base,
                                  weighted_avg=c.weighted_avg,)
        case "deploy":
            require_configs(
                cfg,
                ["output_dir", "deploy_feature_dir", "target_label", "cat_labels", "cont_labels", "model_path"], # this one requires the model_path key!
                prefix="modeling"
            )
            c = cfg.modeling
            from .modeling.marugoto.transformer.helpers import deploy_categorical_model_
            deploy_categorical_model_(clini_table=Path(c.clini_table),
                                      slide_table=Path(c.slide_table),
                                      feature_dir=Path(c.deploy_feature_dir),
                                      output_path=Path(c.output_dir),
                                      target_label=c.target_label,
                                      cat_labels=c.cat_labels,
                                      cont_labels=c.cont_labels,
                                      model_path=Path(c.model_path))
        case "statistics":
            require_configs(
                cfg,
                ["pred_csvs", "target_label", "true_class", "output_dir"],
                prefix="modeling.statistics")
            from .modeling.statistics import compute_stats
            c = cfg.modeling.statistics
            if isinstance(c.pred_csvs,str):
                c.pred_csvs = [c.pred_csvs]
            compute_stats(pred_csvs=[Path(x) for x in c.pred_csvs],
                          target_label=c.target_label,
                          true_class=c.true_class,
                          output_dir=Path(c.output_dir))
        case "heatmaps":
            require_configs(
                cfg,
                ["feature_dir","wsi_dir","model_path","output_dir", "n_toptiles"], 
                prefix="heatmaps")
            c = cfg.heatmaps
            from .heatmaps.__main__ import main
            main(slide_name=str(c.slide_name),
                 feature_dir=Path(c.feature_dir),
                 wsi_dir=Path(c.wsi_dir),
                 model_path=Path(c.model_path),
                 output_dir=Path(c.output_dir),
                 n_toptiles=int(c.n_toptiles),
                 overview=c.overview)
        case _:
            raise ConfigurationError(f"Unknown command {args.command}")

def main() -> None:
    parser = argparse.ArgumentParser(prog="stamp", description="STAMP: Solid Tumor Associative Modeling in Pathology")
    parser.add_argument("--config", "-c", type=Path, default=None, help=f"Path to config file (if unspecified, defaults to {DEFAULT_CONFIG_FILE.absolute()} or the default STAMP config file shipped with the package if {DEFAULT_CONFIG_FILE.absolute()} does not exist)")

    commands = parser.add_subparsers(dest="command")
    commands.add_parser("init", help="Create a new STAMP configuration file at the path specified by --config")
    commands.add_parser("setup", help="Download required resources")
    commands.add_parser("preprocess", help="Preprocess whole-slide images into feature vectors")
    commands.add_parser("train", help="Train a Vision Transformer model")
    commands.add_parser("crossval", help="Train a Vision Transformer model with cross validation for modeling.n_splits folds")
    commands.add_parser("deploy", help="Deploy a trained Vision Transformer model")
    commands.add_parser("statistics", help="Generate AUROCs and AUPRCs with 95%%CI for a trained Vision Transformer model")
    commands.add_parser("config", help="Print the loaded configuration")
    commands.add_parser("heatmaps", help="Generate heatmaps for a trained model")

    args = parser.parse_args()

    # If no command is given, print help and exit
    if args.command is None:
        parser.print_help()
        exit(1)

    # Run the CLI
    try:
        run_cli(args)
    except ConfigurationError as e:
        print(e)
        exit(1)

if __name__ == "__main__":
    main()
