#!/usr/bin/env python3
"""
Script to check if nilmtk and nilmtk-contrib are installed in the current environment
and provide interactive installation options.
"""

import importlib.util
import subprocess
import sys

def check_package(package_name):
    """Check if a package is installed using importlib.util.find_spec() - much faster than importing."""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print(f"✓ {package_name} is installed")
        return True
    else:
        print(f"✗ {package_name} is NOT installed")
        return False

def install_package(package_name, description=""):
    """Install a package using pip."""
    try:
        print(f"\n📦 Installing {package_name}{description}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}")
        print(f"Error: {e.stderr}")
        return False

def get_user_choice():
    """Get user's installation preference."""
    print("\n" + "=" * 60)
    print("🔧 INSTALLATION OPTIONS")
    print("=" * 60)
    print("\nChoose what you want to install:")
    print("1. Install ALL missing packages")
    print("2. Install only Core NILM packages (nilmtk, nilmtk-contrib)")
    print("3. Install only Data Science essentials (pandas, numpy, scipy, matplotlib)")
    print("4. Install only Jupyter essentials (ipykernel, notebook)")
    print("5. Install only missing visualization packages (seaborn, plotly)")
    print("6. Custom selection (choose individual packages)")
    print("7. Skip installation (just show the report)")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7']:
                return int(choice)
            else:
                print("❌ Invalid choice. Please enter a number between 0 and 7.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def get_custom_selection(missing_packages):
    """Let user select individual packages to install."""
    if not missing_packages:
        print("✅ No missing packages to install!")
        return []
    
    print("\n📦 Available missing packages:")
    for i, pkg in enumerate(missing_packages, 1):
        print(f"{i}. {pkg}")
    
    print("\nEnter the numbers of packages you want to install (comma-separated, e.g., '1,3,5'):")
    print("Or enter 'all' to install all missing packages")
    
    while True:
        selection = input("Your choice: ").strip().lower()
        
        if selection == 'all':
            return missing_packages
        
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected = []
            for idx in indices:
                if 0 <= idx < len(missing_packages):
                    selected.append(missing_packages[idx])
                else:
                    print(f"❌ Invalid number: {idx + 1}")
                    break
            else:
                return selected
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas.")

def main():
    """Main function to check required packages."""
    print("Checking NILM and Data Science environment...")
    print("=" * 60)
    
    # Core NILM packages
    print("\n🔍 Core NILM Packages:")
    print("-" * 30)
    nilmtk_installed = check_package("nilmtk")
    nilmtk_contrib_installed = check_package("nilmtk_contrib")
    
    # Data processing libraries
    print("\n📊 Data Processing Libraries:")
    print("-" * 30)
    pandas_installed = check_package("pandas")
    numpy_installed = check_package("numpy")
    scipy_installed = check_package("scipy")
    
    # Machine learning libraries
    print("\n🤖 Machine Learning Libraries:")
    print("-" * 30)
    sklearn_installed = check_package("sklearn")
    tensorflow_installed = check_package("tensorflow")
    torch_installed = check_package("torch")
    
    # Visualization libraries
    print("\n📈 Visualization Libraries:")
    print("-" * 30)
    matplotlib_installed = check_package("matplotlib")
    seaborn_installed = check_package("seaborn")
    plotly_installed = check_package("plotly")
    
    # Utility libraries
    print("\n🛠️  Utility Libraries:")
    print("-" * 30)
    jupyter_installed = check_package("jupyter")
    tqdm_installed = check_package("tqdm")
    joblib_installed = check_package("joblib")
    
    # Jupyter-specific packages (essential for notebook functionality)
    print("\n📓 Jupyter Environment:")
    print("-" * 30)
    ipykernel_installed = check_package("ipykernel")
    notebook_installed = check_package("notebook")
    jupyterlab_installed = check_package("jupyterlab")
    ipywidgets_installed = check_package("ipywidgets")
    
    print("\n" + "=" * 60)
    print("\n📋 Summary:")
    print("-" * 30)
    
    # Check core NILM packages
    if nilmtk_installed and nilmtk_contrib_installed:
        print("✅ Core NILM packages: COMPLETE")
    elif nilmtk_installed:
        print("⚠️  Core NILM packages: PARTIAL (missing nilmtk-contrib)")
    elif nilmtk_contrib_installed:
        print("⚠️  Core NILM packages: PARTIAL (missing nilmtk)")
    else:
        print("❌ Core NILM packages: MISSING")
    
    # Check data science essentials
    data_science_essentials = [pandas_installed, numpy_installed, scipy_installed, matplotlib_installed]
    if all(data_science_essentials):
        print("✅ Data Science essentials: COMPLETE")
    else:
        missing_essentials = []
        if not pandas_installed: missing_essentials.append("pandas")
        if not numpy_installed: missing_essentials.append("numpy")
        if not scipy_installed: missing_essentials.append("scipy")
        if not matplotlib_installed: missing_essentials.append("matplotlib")
        print(f"⚠️  Data Science essentials: PARTIAL (missing: {', '.join(missing_essentials)})")
    
    # Check Jupyter environment
    jupyter_core = [jupyter_installed, ipykernel_installed]
    if all(jupyter_core):
        print("✅ Jupyter core: COMPLETE")
    else:
        missing_jupyter_core = []
        if not jupyter_installed: missing_jupyter_core.append("jupyter")
        if not ipykernel_installed: missing_jupyter_core.append("ipykernel")
        print(f"⚠️  Jupyter core: PARTIAL (missing: {', '.join(missing_jupyter_core)})")
        
    # Check Jupyter notebook functionality
    if ipykernel_installed and notebook_installed:
        print("✅ Jupyter notebook functionality: READY")
    elif ipykernel_installed:
        print("⚠️  Jupyter notebook functionality: PARTIAL (missing notebook interface)")
    elif notebook_installed:
        print("⚠️  Jupyter notebook functionality: BROKEN (missing ipykernel - CANNOT run Python code!)")
    else:
        print("❌ Jupyter notebook functionality: MISSING")
    
    # Installation suggestions
    print("\n💡 Installation Suggestions:")
    print("-" * 30)
    
    if not nilmtk_installed:
        print("• pip install nilmtk")
    if not nilmtk_contrib_installed:
        print("• pip install nilmtk-contrib")
    
    missing_ds = []
    if not pandas_installed: missing_ds.append("pandas")
    if not numpy_installed: missing_ds.append("numpy")
    if not scipy_installed: missing_ds.append("scipy")
    if not matplotlib_installed: missing_ds.append("matplotlib")
    if not sklearn_installed: missing_ds.append("scikit-learn")
    if not seaborn_installed: missing_ds.append("seaborn")
    if not tqdm_installed: missing_ds.append("tqdm")
    if not joblib_installed: missing_ds.append("joblib")
    
    if missing_ds:
        print(f"• pip install {' '.join(missing_ds)}")
    
    if not tensorflow_installed:
        print("• pip install tensorflow (for deep learning)")
    if not torch_installed:
        print("• pip install torch (for deep learning)")
    if not plotly_installed:
        print("• pip install plotly (for interactive plots)")
    if not ipykernel_installed:
        print("• pip install ipykernel (REQUIRED for Jupyter to run Python code)")
    if not notebook_installed:
        print("• pip install notebook (classic Jupyter notebook interface)")
    if not jupyterlab_installed:
        print("• pip install jupyterlab (modern JupyterLab interface)")
    if not ipywidgets_installed:
        print("• pip install ipywidgets (for interactive widgets)")
    
    print("\n🎯 Quick install command for common packages:")
    print("pip install pandas numpy scipy matplotlib scikit-learn seaborn tqdm joblib")
    
    print("\n📓 Essential Jupyter setup:")
    print("pip install ipykernel notebook jupyterlab")
    
    print("\n🚀 Complete Jupyter environment:")
    print("pip install ipykernel notebook jupyterlab ipywidgets")

def main():
    """Main function to check required packages and handle installation."""
    print("Checking NILM and Data Science environment...")
    print("=" * 60)
    
    # Core NILM packages
    print("\n🔍 Core NILM Packages:")
    print("-" * 30)
    nilmtk_installed = check_package("nilmtk")
    nilmtk_contrib_installed = check_package("nilmtk_contrib")
    
    # Data processing libraries
    print("\n📊 Data Processing Libraries:")
    print("-" * 30)
    pandas_installed = check_package("pandas")
    numpy_installed = check_package("numpy")
    scipy_installed = check_package("scipy")
    
    # Machine learning libraries
    print("\n🤖 Machine Learning Libraries:")
    print("-" * 30)
    sklearn_installed = check_package("sklearn")
    tensorflow_installed = check_package("tensorflow")
    torch_installed = check_package("torch")
    
    # Visualization libraries
    print("\n📈 Visualization Libraries:")
    print("-" * 30)
    matplotlib_installed = check_package("matplotlib")
    seaborn_installed = check_package("seaborn")
    plotly_installed = check_package("plotly")
    
    # Utility libraries
    print("\n🛠️  Utility Libraries:")
    print("-" * 30)
    jupyter_installed = check_package("jupyter")
    tqdm_installed = check_package("tqdm")
    joblib_installed = check_package("joblib")
    
    # Jupyter-specific packages (essential for notebook functionality)
    print("\n📓 Jupyter Environment:")
    print("-" * 30)
    ipykernel_installed = check_package("ipykernel")
    notebook_installed = check_package("notebook")
    jupyterlab_installed = check_package("jupyterlab")
    ipywidgets_installed = check_package("ipywidgets")
    
    # Collect all missing packages
    missing_packages = []
    if not nilmtk_installed: missing_packages.append("nilmtk")
    if not nilmtk_contrib_installed: missing_packages.append("nilmtk-contrib")
    if not pandas_installed: missing_packages.append("pandas")
    if not numpy_installed: missing_packages.append("numpy")
    if not scipy_installed: missing_packages.append("scipy")
    if not sklearn_installed: missing_packages.append("scikit-learn")
    if not tensorflow_installed: missing_packages.append("tensorflow")
    if not torch_installed: missing_packages.append("torch")
    if not matplotlib_installed: missing_packages.append("matplotlib")
    if not seaborn_installed: missing_packages.append("seaborn")
    if not plotly_installed: missing_packages.append("plotly")
    if not jupyter_installed: missing_packages.append("jupyter")
    if not tqdm_installed: missing_packages.append("tqdm")
    if not joblib_installed: missing_packages.append("joblib")
    if not ipykernel_installed: missing_packages.append("ipykernel")
    if not notebook_installed: missing_packages.append("notebook")
    if not jupyterlab_installed: missing_packages.append("jupyterlab")
    if not ipywidgets_installed: missing_packages.append("ipywidgets")
    
    print("\n" + "=" * 60)
    print("\n📋 Summary:")
    print("-" * 30)
    
    # Check core NILM packages
    if nilmtk_installed and nilmtk_contrib_installed:
        print("✅ Core NILM packages: COMPLETE")
    elif nilmtk_installed:
        print("⚠️  Core NILM packages: PARTIAL (missing nilmtk-contrib)")
    elif nilmtk_contrib_installed:
        print("⚠️  Core NILM packages: PARTIAL (missing nilmtk)")
    else:
        print("❌ Core NILM packages: MISSING")
    
    # Check data science essentials
    data_science_essentials = [pandas_installed, numpy_installed, scipy_installed, matplotlib_installed]
    if all(data_science_essentials):
        print("✅ Data Science essentials: COMPLETE")
    else:
        missing_essentials = []
        if not pandas_installed: missing_essentials.append("pandas")
        if not numpy_installed: missing_essentials.append("numpy")
        if not scipy_installed: missing_essentials.append("scipy")
        if not matplotlib_installed: missing_essentials.append("matplotlib")
        print(f"⚠️  Data Science essentials: PARTIAL (missing: {', '.join(missing_essentials)})")
    
    # Check Jupyter environment
    jupyter_core = [jupyter_installed, ipykernel_installed]
    if all(jupyter_core):
        print("✅ Jupyter core: COMPLETE")
    else:
        missing_jupyter_core = []
        if not jupyter_installed: missing_jupyter_core.append("jupyter")
        if not ipykernel_installed: missing_jupyter_core.append("ipykernel")
        print(f"⚠️  Jupyter core: PARTIAL (missing: {', '.join(missing_jupyter_core)})")
        
    # Check Jupyter notebook functionality
    if ipykernel_installed and notebook_installed:
        print("✅ Jupyter notebook functionality: READY")
    elif ipykernel_installed:
        print("⚠️  Jupyter notebook functionality: PARTIAL (missing notebook interface)")
    elif notebook_installed:
        print("⚠️  Jupyter notebook functionality: BROKEN (missing ipykernel - CANNOT run Python code!)")
    else:
        print("❌ Jupyter notebook functionality: MISSING")
    
    # Show missing packages count
    if missing_packages:
        print(f"\n📦 Total missing packages: {len(missing_packages)}")
        print(f"   {', '.join(missing_packages)}")
    else:
        print("\n🎉 All packages are installed!")
        return
    
    # Get user's installation choice
    choice = get_user_choice()
    
    # Handle installation based on user choice
    packages_to_install = []
    
    if choice == 0:  # Exit
        print("👋 Goodbye!")
        return
    elif choice == 1:  # Install all missing
        packages_to_install = missing_packages
    elif choice == 2:  # Core NILM only
        if not nilmtk_installed: packages_to_install.append("nilmtk")
        if not nilmtk_contrib_installed: packages_to_install.append("nilmtk-contrib")
    elif choice == 3:  # Data Science essentials only
        if not pandas_installed: packages_to_install.append("pandas")
        if not numpy_installed: packages_to_install.append("numpy")
        if not scipy_installed: packages_to_install.append("scipy")
        if not matplotlib_installed: packages_to_install.append("matplotlib")
    elif choice == 4:  # Jupyter essentials only
        if not ipykernel_installed: packages_to_install.append("ipykernel")
        if not notebook_installed: packages_to_install.append("notebook")
    elif choice == 5:  # Visualization only
        if not seaborn_installed: packages_to_install.append("seaborn")
        if not plotly_installed: packages_to_install.append("plotly")
    elif choice == 6:  # Custom selection
        packages_to_install = get_custom_selection(missing_packages)
    elif choice == 7:  # Skip installation
        print("\n💡 Installation skipped. Here are the manual installation commands:")
        print("-" * 50)
        for pkg in missing_packages:
            print(f"• pip install {pkg}")
        return
    
    # Install selected packages
    if packages_to_install:
        print(f"\n🚀 Installing {len(packages_to_install)} package(s)...")
        print("-" * 50)
        
        successful_installs = []
        failed_installs = []
        
        for package in packages_to_install:
            if install_package(package):
                successful_installs.append(package)
            else:
                failed_installs.append(package)
        
        # Installation summary
        print("\n" + "=" * 60)
        print("📊 INSTALLATION SUMMARY")
        print("=" * 60)
        
        if successful_installs:
            print(f"\n✅ Successfully installed: {len(successful_installs)} package(s)")
            for pkg in successful_installs:
                print(f"   • {pkg}")
        
        if failed_installs:
            print(f"\n❌ Failed to install: {len(failed_installs)} package(s)")
            for pkg in failed_installs:
                print(f"   • {pkg}")
            print("\n💡 You can try installing these manually:")
            for pkg in failed_installs:
                print(f"   pip install {pkg}")
        
        if successful_installs:
            print(f"\n🎉 Installation complete! You may need to restart your Python environment.")
    else:
        print("\n✅ No packages selected for installation.")

if __name__ == "__main__":
    main()
