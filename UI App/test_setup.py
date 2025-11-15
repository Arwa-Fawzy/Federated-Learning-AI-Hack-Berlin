"""
Test script to verify UI App setup and data availability
Run this before launching the dashboard
"""

import sys
from pathlib import Path
import importlib.util

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing Package Imports...")
    print("=" * 60)
    
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly'
    }
    
    all_passed = True
    for package_name, import_name in required_packages.items():
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                print(f"✅ {package_name:15s} - OK")
            else:
                print(f"❌ {package_name:15s} - NOT FOUND")
                all_passed = False
        except (ImportError, ModuleNotFoundError):
            print(f"❌ {package_name:15s} - NOT FOUND")
            all_passed = False
    
    print()
    return all_passed

def test_data_files():
    """Test if required data files exist"""
    print("=" * 60)
    print("Testing Data Files...")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "federated_data" / "hybrid"
    
    print(f"Looking in: {data_dir}")
    print()
    
    required_files = [
        "client_metadata.json",
        "client_0.csv",
        "client_1.csv",
        "client_2.csv",
        "client_3.csv",
        "client_4.csv"
    ]
    
    all_passed = True
    for filename in required_files:
        file_path = data_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {filename:25s} - OK ({size_mb:.2f} MB)")
        else:
            print(f"❌ {filename:25s} - NOT FOUND")
            all_passed = False
    
    print()
    return all_passed

def test_utils():
    """Test if utils.py functions work"""
    print("=" * 60)
    print("Testing Utility Functions...")
    print("=" * 60)
    
    try:
        from utils import validate_data_files, load_metadata, load_client_data
        
        # Test data validation
        valid, missing = validate_data_files()
        if valid:
            print("✅ Data validation - OK")
        else:
            print(f"❌ Data validation - FAILED")
            print(f"   Missing files: {missing}")
            return False
        
        # Test metadata loading
        try:
            metadata = load_metadata()
            print(f"✅ Load metadata - OK ({metadata['n_clients']} clients)")
        except Exception as e:
            print(f"❌ Load metadata - FAILED: {str(e)}")
            return False
        
        # Test client data loading
        try:
            data = load_client_data(0)
            print(f"✅ Load client data - OK ({len(data):,} samples)")
        except Exception as e:
            print(f"❌ Load client data - FAILED: {str(e)}")
            return False
        
        print()
        return True
        
    except ImportError as e:
        print(f"❌ Import utils - FAILED: {str(e)}")
        print()
        return False

def test_app_file():
    """Test if app.py exists and has no syntax errors"""
    print("=" * 60)
    print("Testing App File...")
    print("=" * 60)
    
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print("❌ app.py - NOT FOUND")
        print()
        return False
    
    print(f"✅ app.py - EXISTS")
    
    # Try to compile (syntax check)
    try:
        with open(app_path, 'r', encoding='utf-8') as f:
            compile(f.read(), app_path, 'exec')
        print("✅ app.py - NO SYNTAX ERRORS")
        print()
        return True
    except SyntaxError as e:
        print(f"❌ app.py - SYNTAX ERROR: {str(e)}")
        print()
        return False

def main():
    """Run all tests"""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "SenorMatics Dashboard Setup Test" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Data Files", test_data_files()))
    results.append(("Utility Functions", test_utils()))
    results.append(("App File", test_app_file()))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25s}: {status}")
        if not passed:
            all_passed = False
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print()
        print("You can now run the dashboard:")
        print("  streamlit run app.py")
        print()
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print()
        print("Please fix the issues above before running the dashboard.")
        print()
        print("Common fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Generate data: python ../heterogeneous_data_generator.py")
        print("  3. Check file paths in utils.py")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())

