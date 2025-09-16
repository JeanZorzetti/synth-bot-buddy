const fs = require('fs');

// Load the extracted documentation
const docs = JSON.parse(fs.readFileSync('deriv-api-docs.json', 'utf8'));

console.log('🔍 ANALYZING DERIV API DOCUMENTATION FOR BOT IMPLEMENTATION\n');

// Critical sections for bot development
const criticalSections = {
    authentication: [],
    trading: [],
    websockets: [],
    contracts: [],
    marketData: [],
    accountManagement: [],
    errorHandling: []
};

// Key terms to search for
const keyTerms = {
    authentication: ['oauth', 'api_token', 'authorize', 'app_id', 'authentication'],
    trading: ['buy', 'sell', 'proposal', 'contract', 'trade', 'multiplier', 'accumulator'],
    websockets: ['websocket', 'subscribe', 'stream', 'real-time', 'connection'],
    contracts: ['contract_type', 'symbol', 'duration', 'amount', 'barrier'],
    marketData: ['ticks', 'price', 'symbol', 'market', 'active_symbols'],
    accountManagement: ['balance', 'statement', 'portfolio', 'profit_table'],
    errorHandling: ['error', 'error_code', 'invalid', 'failed']
};

// Extract critical information
Object.entries(docs.content).forEach(([pageName, pageData]) => {
    const content = pageData.mainContent.toLowerCase();
    
    // Categorize content based on key terms
    Object.entries(keyTerms).forEach(([category, terms]) => {
        if (terms.some(term => content.includes(term))) {
            criticalSections[category].push({
                page: pageName,
                url: pageData.url,
                relevantContent: pageData.mainContent,
                codeBlocks: pageData.codeBlocks || [],
                headings: pageData.headings || []
            });
        }
    });
});

// Generate comprehensive bot implementation guide
let guide = `# DERIV API COMPLETE BOT IMPLEMENTATION GUIDE\n\n`;
guide += `📊 **Documentation Analysis:**\n`;
guide += `- Total pages analyzed: ${Object.keys(docs.content).length}\n`;
guide += `- Total content: ${docs.summary.totalCharacters.toLocaleString()} characters\n`;
guide += `- Scraped: ${docs.scrapedAt}\n\n`;

// 1. AUTHENTICATION SECTION
guide += `## 🔐 1. AUTHENTICATION & SETUP\n\n`;
if (criticalSections.authentication.length > 0) {
    criticalSections.authentication.forEach(section => {
        guide += `### ${section.page}\n`;
        guide += `**URL:** ${section.url}\n\n`;
        
        // Extract code examples
        if (section.codeBlocks.length > 0) {
            guide += `**Code Examples:**\n`;
            section.codeBlocks.forEach((code, i) => {
                guide += `\`\`\`${code.language}\n${code.content}\n\`\`\`\n\n`;
            });
        }
        
        // Extract key headings
        if (section.headings.length > 0) {
            guide += `**Key Topics:**\n`;
            section.headings.forEach(h => guide += `- ${h.text}\n`);
            guide += `\n`;
        }
        
        guide += `**Content Summary:**\n${section.relevantContent.substring(0, 1000)}...\n\n`;
        guide += `---\n\n`;
    });
}

// 2. WEBSOCKET CONNECTION
guide += `## 🔌 2. WEBSOCKET CONNECTION\n\n`;
if (criticalSections.websockets.length > 0) {
    criticalSections.websockets.forEach(section => {
        guide += `### ${section.page}\n`;
        guide += `**URL:** ${section.url}\n\n`;
        
        if (section.codeBlocks.length > 0) {
            guide += `**Implementation Examples:**\n`;
            section.codeBlocks.forEach((code, i) => {
                guide += `\`\`\`${code.language}\n${code.content}\n\`\`\`\n\n`;
            });
        }
        
        guide += `**Details:**\n${section.relevantContent.substring(0, 1000)}...\n\n`;
        guide += `---\n\n`;
    });
}

// 3. TRADING OPERATIONS
guide += `## 📈 3. TRADING OPERATIONS\n\n`;
if (criticalSections.trading.length > 0) {
    criticalSections.trading.forEach(section => {
        guide += `### ${section.page}\n`;
        guide += `**URL:** ${section.url}\n\n`;
        
        if (section.codeBlocks.length > 0) {
            guide += `**Trading Examples:**\n`;
            section.codeBlocks.forEach((code, i) => {
                guide += `\`\`\`${code.language}\n${code.content}\n\`\`\`\n\n`;
            });
        }
        
        if (section.headings.length > 0) {
            guide += `**Available Operations:**\n`;
            section.headings.forEach(h => guide += `- ${h.text}\n`);
            guide += `\n`;
        }
        
        guide += `**Implementation Details:**\n${section.relevantContent.substring(0, 1200)}...\n\n`;
        guide += `---\n\n`;
    });
}

// 4. MARKET DATA
guide += `## 📊 4. MARKET DATA & SYMBOLS\n\n`;
if (criticalSections.marketData.length > 0) {
    criticalSections.marketData.forEach(section => {
        guide += `### ${section.page}\n`;
        guide += `**URL:** ${section.url}\n\n`;
        
        if (section.codeBlocks.length > 0) {
            guide += `**Data Fetching Examples:**\n`;
            section.codeBlocks.forEach((code, i) => {
                guide += `\`\`\`${code.language}\n${code.content}\n\`\`\`\n\n`;
            });
        }
        
        guide += `**Content:**\n${section.relevantContent.substring(0, 1000)}...\n\n`;
        guide += `---\n\n`;
    });
}

// 5. ACCOUNT MANAGEMENT
guide += `## 👤 5. ACCOUNT MANAGEMENT\n\n`;
if (criticalSections.accountManagement.length > 0) {
    criticalSections.accountManagement.forEach(section => {
        guide += `### ${section.page}\n`;
        guide += `**URL:** ${section.url}\n\n`;
        
        if (section.codeBlocks.length > 0) {
            guide += `**Account Operations:**\n`;
            section.codeBlocks.forEach((code, i) => {
                guide += `\`\`\`${code.language}\n${code.content}\n\`\`\`\n\n`;
            });
        }
        
        guide += `**Details:**\n${section.relevantContent.substring(0, 1000)}...\n\n`;
        guide += `---\n\n`;
    });
}

// 6. ERROR HANDLING
guide += `## ⚠️ 6. ERROR HANDLING\n\n`;
if (criticalSections.errorHandling.length > 0) {
    criticalSections.errorHandling.forEach(section => {
        guide += `### ${section.page}\n`;
        guide += `**URL:** ${section.url}\n\n`;
        
        if (section.codeBlocks.length > 0) {
            guide += `**Error Handling Examples:**\n`;
            section.codeBlocks.forEach((code, i) => {
                guide += `\`\`\`${code.language}\n${code.content}\n\`\`\`\n\n`;
            });
        }
        
        guide += `**Error Information:**\n${section.relevantContent.substring(0, 1000)}...\n\n`;
        guide += `---\n\n`;
    });
}

// SUMMARY STATISTICS
guide += `\n## 📈 IMPLEMENTATION STATISTICS\n\n`;
Object.entries(criticalSections).forEach(([category, sections]) => {
    guide += `- **${category.toUpperCase()}:** ${sections.length} relevant pages found\n`;
});

guide += `\n## 🎯 NEXT STEPS FOR BOT IMPLEMENTATION\n\n`;
guide += `1. **Setup Authentication:** Implement OAuth or API token authentication\n`;
guide += `2. **Establish WebSocket Connection:** Create persistent connection for real-time data\n`;
guide += `3. **Implement Trading Logic:** Add buy/sell functionality with proper contract handling\n`;
guide += `4. **Add Market Data Streaming:** Subscribe to real-time price feeds\n`;
guide += `5. **Account Management:** Implement balance tracking and portfolio management\n`;
guide += `6. **Error Handling:** Add comprehensive error handling and logging\n`;
guide += `7. **Testing:** Implement thorough testing with demo accounts\n`;
guide += `8. **Risk Management:** Add position sizing and stop-loss functionality\n\n`;

guide += `## 🔗 CRITICAL API ENDPOINTS DISCOVERED\n\n`;

// Extract all unique API endpoints mentioned
const apiEndpoints = new Set();
Object.values(docs.content).forEach(page => {
    const content = page.mainContent;
    // Look for common API patterns
    const endpoints = content.match(/\b(authorize|buy|sell|proposal|ticks|balance|statement|portfolio|active_symbols)\b/gi);
    if (endpoints) {
        endpoints.forEach(endpoint => apiEndpoints.add(endpoint.toLowerCase()));
    }
});

Array.from(apiEndpoints).sort().forEach(endpoint => {
    guide += `- \`${endpoint}\`\n`;
});

guide += `\n---\n\n**Total comprehensive documentation extracted and analyzed!**\n`;
guide += `**Ready for bot implementation with complete API knowledge.**\n`;

// Save the comprehensive guide
fs.writeFileSync('DERIV-BOT-IMPLEMENTATION-GUIDE.md', guide);
console.log('✅ Comprehensive bot implementation guide created!');
console.log(`📄 File: DERIV-BOT-IMPLEMENTATION-GUIDE.md`);
console.log(`📊 Guide size: ${Math.round(Buffer.byteLength(guide, 'utf8') / 1024)}KB`);

// Also create a quick reference JSON
const quickRef = {
    criticalEndpoints: Array.from(apiEndpoints).sort(),
    documentationSections: Object.keys(criticalSections).map(key => ({
        category: key,
        pagesFound: criticalSections[key].length,
        pages: criticalSections[key].map(s => ({ title: s.page, url: s.url }))
    })),
    totalPages: Object.keys(docs.content).length,
    extractionDate: docs.scrapedAt
};

fs.writeFileSync('deriv-api-quick-reference.json', JSON.stringify(quickRef, null, 2));
console.log('📋 Quick reference created: deriv-api-quick-reference.json');

console.log(`\n🎯 ANALYSIS COMPLETE - BOT READY FOR IMPLEMENTATION!`);