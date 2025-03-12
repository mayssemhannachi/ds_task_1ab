from typing import List, Dict, Any
import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, Index
import nltk
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from difflib import get_close_matches

class ProductRecommendationService:
    # Common product-related terms and their variations
    PRODUCT_TERMS = {
        # Synonyms and related terms
        'cup': ['mug', 'tumbler', 'glass', 'beaker'],
        'plate': ['dish', 'platter', 'saucer'],
        'bowl': ['basin', 'dish', 'container'],
        'kitchen': ['cooking', 'culinary', 'dining'],
        'garden': ['outdoor', 'patio', 'yard'],
        'decoration': ['ornament', 'decor', 'accessory'],
        'gift': ['present', 'souvenir', 'keepsake'],
        'box': ['container', 'case', 'storage'],
        'bag': ['pouch', 'sack', 'tote'],
        'bottle': ['container', 'flask', 'jar'],
        'toy': ['game', 'plaything', 'entertainment'],
        'jewelry': ['accessory', 'ornament', 'decoration'],
        'cloth': ['fabric', 'textile', 'material'],
        'light': ['lamp', 'lantern', 'illumination'],
        'chair': ['seat', 'stool', 'furniture'],
        
        # Common misspellings
        'ceramic': ['ceramik', 'ceramics', 'ceremic'],
        'jewelry': ['jewelery', 'jewellry'],
        'accessories': ['accessorise', 'accessorize'],
        'kitchen': ['kichen', 'kitchin'],
        'christmas': ['xmas', 'christmass'],
        'decoration': ['decor', 'decoracion'],
        'birthday': ['bday', 'birthay'],
        'beautiful': ['beutiful', 'beautifull'],
        'garden': ['graden', 'gadern'],
        'flower': ['flour', 'flwr'],
    }

    def __init__(self, api_key: str, environment: str = "gcp-starter", index_name: str = "product-embeddings", top_k: int = 10):
        """
        Initialize the Product Recommendation Service
        
        Args:
            api_key (str): Pinecone API key
            environment (str): Pinecone environment
            index_name (str): Name of the Pinecone index
            top_k (int): Number of recommendations to return
        """
        # Initialize Pinecone
        self.pinecone = Pinecone(api_key=api_key, environment=environment)
        self.index = self.pinecone.Index(index_name)
        
        # Initialize the embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.top_k = top_k
        
        # Initialize NLTK components
        self._initialize_nltk()
        
    def _initialize_nltk(self):
        """Initialize NLTK and download required resources"""
        try:
            # Try to tokenize a sample text to check if punkt is available
            word_tokenize("test sentence")
        except LookupError:
            try:
                nltk.download('punkt')
            except Exception as e:
                logging.warning(f"Could not download punkt: {str(e)}")
        
        try:
            # Try to access WordNet
            wordnet.synsets('test')
        except LookupError:
            try:
                nltk.download('wordnet')
                nltk.download('omw-1.4')
            except Exception as e:
                logging.warning(f"Could not download wordnet: {str(e)}")
    
    def _tokenize_safely(self, text: str) -> List[str]:
        """Safely tokenize text, falling back to simple split if NLTK fails"""
        try:
            return word_tokenize(text.lower())
        except Exception as e:
            logging.warning(f"NLTK tokenization failed: {str(e)}")
            return text.lower().split()

    def _get_wordnet_synonyms(self, word: str) -> List[str]:
        """Safely get synonyms from WordNet"""
        try:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and len(synonym) > 2:
                        synonyms.add(synonym)
            return list(synonyms)
        except Exception as e:
            logging.warning(f"WordNet lookup failed: {str(e)}")
            return []

    def _sanitize_query(self, query: str) -> str:
        """Remove any potentially harmful characters and standardize the query"""
        query = re.sub(r'[^\w\s.,?!-]', '', query)
        return query.strip()
    
    def _validate_query(self, query: str) -> bool:
        """Validate if the query is appropriate for processing"""
        if not query or len(query.strip()) < 2:
            return False
        return True

    def _get_product_variations(self, word: str) -> List[str]:
        """Get predefined product variations including synonyms and common misspellings"""
        variations = set()
        word_lower = word.lower()
        
        # Check predefined variations
        if word_lower in self.PRODUCT_TERMS:
            variations.update(self.PRODUCT_TERMS[word_lower])
        
        try:
            # Add common misspellings using fuzzy matching
            close_matches = get_close_matches(word_lower, self.PRODUCT_TERMS.keys(), n=3, cutoff=0.8)
            for match in close_matches:
                if match != word_lower:
                    variations.add(match)
                    variations.update(self.PRODUCT_TERMS.get(match, []))
        except Exception as e:
            logging.warning(f"Error in fuzzy matching: {str(e)}")
            # Continue without fuzzy matches if there's an error
            pass
        
        return list(variations)

    def _expand_query(self, query: str) -> List[str]:
        """
        Expand the query to include variations, synonyms, related terms, and common misspellings
        Example: "ceramic cup" -> ["ceramic cup", "ceramic mug", "ceramik cup", "pottery cup", ...]
        """
        queries = [query]
        words = self._tokenize_safely(query)
        expanded_words = {}

        for word in words:
            word_variations = set()
            
            # 1. Add base word
            word_variations.add(word)
            
            # 2. Add singular/plural variations
            if word.endswith('s'):
                word_variations.add(word[:-1])  # Remove 's'
            else:
                word_variations.add(word + 's')  # Add 's'
            
            # Handle 'y' to 'ies'
            if word.endswith('y'):
                word_variations.add(word[:-1] + 'ies')
            elif word.endswith('ies'):
                word_variations.add(word[:-3] + 'y')
            
            # 3. Add WordNet synonyms
            word_variations.update(self._get_wordnet_synonyms(word))
            
            # 4. Add product-specific variations (synonyms, related terms, misspellings)
            word_variations.update(self._get_product_variations(word))
            
            expanded_words[word] = list(word_variations)

        # Generate combinations of variations
        for word in words:
            variations = expanded_words[word]
            for variation in variations:
                new_query = query
                for w in words:
                    if w == word:
                        new_query = new_query.replace(w, variation)
                if new_query != query:
                    queries.append(new_query)

        # Remove duplicates while preserving order
        seen = set()
        expanded_queries = []
        for q in queries:
            if q not in seen:
                expanded_queries.append(q)
                seen.add(q)

        return expanded_queries
    
    def _format_response(self, matches: List[Dict], query: str) -> Dict[str, Any]:
        """Format the response with products and natural language description"""
        products = []
        for match in matches:
            product = {
                'stock_code': match['metadata']['stock_code'],
                'description': match['metadata']['description'],
                'country': match['metadata']['country'],
                'unit_price': match['metadata']['unit_price'],
                'similarity_score': float(match['score'])
            }
            products.append(product)
        
        # Create natural language response
        if products:
            response_text = f"Found {len(products)} products matching your query '{query}'. "
            response_text += f"The top match is {products[0]['description']} "
            response_text += f"from {products[0]['country']}."
        else:
            response_text = f"No products found matching your query '{query}'."
        
        return {
            'query': query,
            'products': products,
            'response_text': response_text
        }
    
    def get_recommendations(self, query: str) -> Dict[str, Any]:
        """
        Get product recommendations from a natural language query
        
        Args:
            query (str): Natural language query string
            
        Returns:
            Dict containing:
                - query: Original query
                - products: List of recommended products with metadata
                - response_text: Natural language response
        """
        # Sanitize and validate query
        clean_query = self._sanitize_query(query)
        if not self._validate_query(clean_query):
            return {
                'query': query,
                'products': [],
                'response_text': 'Invalid query. Please provide a more specific search term.'
            }
        
        try:
            # Expand query to include variations
            expanded_queries = self._expand_query(clean_query)
            all_matches = []
            seen_stock_codes = set()
            
            # Search for each query variation
            for expanded_query in expanded_queries:
                # Convert query to vector
                query_vector = self.model.encode(expanded_query).tolist()
                
                # Search in Pinecone
                search_results = self.index.query(
                    vector=query_vector,
                    top_k=self.top_k,
                    include_metadata=True
                )
                
                # Add unique matches
                for match in search_results['matches']:
                    stock_code = match['metadata']['stock_code']
                    if stock_code not in seen_stock_codes:
                        all_matches.append(match)
                        seen_stock_codes.add(stock_code)
            
            # Sort all matches by similarity score
            all_matches.sort(key=lambda x: x['score'], reverse=True)
            
            # Format and return results
            return self._format_response(all_matches, clean_query)
            
        except Exception as e:
            logging.error(f"Error in get_recommendations: {str(e)}")
            return {
                'query': query,
                'products': [],
                'response_text': f'An error occurred while processing your request: {str(e)}'
            }

# Example usage
if __name__ == "__main__":
    # Initialize the service
    service = ProductRecommendationService(
        api_key="your-api-key-here"
    )
    
    # Test queries
    test_queries = [
        "white ceramic cup",  # Will find cups, mugs, ceramics variations
        "garden decoration",  # Will find decorations, decor, outdoor items
        "christmas gift",    # Will find gifts, presents, xmas variations
        "kitchen storage"    # Will find storage, containers, cooking related items
    ]
    
    # Run test queries
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = service.get_recommendations(query)
        print(f"Response: {results['response_text']}")
        print("\nTop Products:")
        for i, product in enumerate(results['products'], 1):
            print(f"{i}. {product['description']} (Score: {product['similarity_score']:.3f})") 