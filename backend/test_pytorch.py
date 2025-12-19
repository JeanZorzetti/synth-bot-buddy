"""
Script de teste para verificar instalação do PyTorch
"""
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import torch
    print(f"\n✅ PyTorch instalado!")
    print(f"   Versão: {torch.__version__}")
    print(f"   Localização: {torch.__file__}")
    print(f"   CUDA disponível: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"\n❌ PyTorch NÃO instalado!")
    print(f"   Erro: {e}")
    print(f"\n📦 Pacotes instalados:")
    import pkg_resources
    installed_packages = [pkg.key for pkg in pkg_resources.working_set]
    torch_packages = [pkg for pkg in installed_packages if 'torch' in pkg]
    if torch_packages:
        print(f"   Pacotes com 'torch': {torch_packages}")
    else:
        print(f"   Nenhum pacote com 'torch' encontrado")
