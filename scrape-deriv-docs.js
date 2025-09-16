const puppeteer = require('puppeteer');
const fs = require('fs');

async function scrapeDerivDocs() {
    const browser = await puppeteer.launch({
        headless: false, // Set to true for production
        defaultViewport: null,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    const allContent = {};

    try {
        console.log('Navigating to Deriv API docs...');
        await page.goto('https://developers.deriv.com/docs/understanding-apis', {
            waitUntil: 'networkidle2',
            timeout: 60000
        });

        // Wait for page to load completely
        await new Promise(resolve => setTimeout(resolve, 3000));

        // Extract main navigation links
        console.log('Extracting navigation structure...');
        const navLinks = await page.evaluate(() => {
            const links = [];
            
            // Try different selectors for navigation
            const selectors = [
                'nav a[href*="/docs/"]',
                '.sidebar a[href*="/docs/"]',
                '.menu a[href*="/docs/"]',
                '.navigation a[href*="/docs/"]',
                'a[href*="/docs/"]'
            ];

            for (const selector of selectors) {
                const elements = document.querySelectorAll(selector);
                if (elements.length > 0) {
                    elements.forEach(link => {
                        const href = link.getAttribute('href');
                        const text = link.textContent?.trim();
                        if (href && text && href.includes('/docs/') && !links.find(l => l.href === href)) {
                            links.push({
                                href: href.startsWith('http') ? href : `https://developers.deriv.com${href}`,
                                text: text,
                                category: 'navigation'
                            });
                        }
                    });
                    break;
                }
            }
            
            return links;
        });

        console.log(`Found ${navLinks.length} navigation links`);

        // Key sections we definitely need
        const criticalSections = [
            'understanding-apis',
            'authentication',
            'websocket-api',
            'trading-api',
            'account-management',
            'portfolio-management',
            'market-data',
            'trading-operations',
            'bot-trading',
            'api-guide',
            'getting-started'
        ];

        // Add critical sections to our links if not already present
        criticalSections.forEach(section => {
            const url = `https://developers.deriv.com/docs/${section}`;
            if (!navLinks.find(link => link.href.includes(section))) {
                navLinks.push({
                    href: url,
                    text: section,
                    category: 'critical'
                });
            }
        });

        // Extract content from each page
        for (let i = 0; i < navLinks.length; i++) {
            const link = navLinks[i];
            console.log(`\nProcessing ${i + 1}/${navLinks.length}: ${link.text} (${link.href})`);

            try {
                await page.goto(link.href, {
                    waitUntil: 'networkidle2',
                    timeout: 30000
                });

                await new Promise(resolve => setTimeout(resolve, 2000));

                // Extract all content from the page
                const pageContent = await page.evaluate(() => {
                    // Remove scripts and styles
                    const scripts = document.querySelectorAll('script, style, nav, header, footer');
                    scripts.forEach(el => el.remove());

                    return {
                        title: document.title,
                        url: window.location.href,
                        mainContent: document.body.innerText || '',
                        headings: Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6')).map(h => ({
                            level: h.tagName.toLowerCase(),
                            text: h.textContent?.trim() || ''
                        })),
                        codeBlocks: Array.from(document.querySelectorAll('pre, code')).map(code => ({
                            language: code.className.match(/language-(\w+)/)?.[1] || 'text',
                            content: code.textContent?.trim() || ''
                        })),
                        tables: Array.from(document.querySelectorAll('table')).map(table => 
                            table.outerHTML
                        ),
                        links: Array.from(document.querySelectorAll('a[href]')).map(a => ({
                            text: a.textContent?.trim() || '',
                            href: a.href
                        })).filter(link => link.text && link.href)
                    };
                });

                allContent[link.text] = pageContent;
                console.log(`✓ Extracted ${pageContent.mainContent.length} characters from ${link.text}`);

            } catch (error) {
                console.error(`Error processing ${link.href}:`, error.message);
            }
        }

        // Extract API reference if available
        console.log('\nLooking for API reference sections...');
        const apiReferenceLinks = await page.evaluate(() => {
            return Array.from(document.querySelectorAll('a[href*="api"], a[href*="reference"], a[href*="endpoint"]'))
                .map(link => ({
                    href: link.href,
                    text: link.textContent?.trim()
                }))
                .filter(link => link.text && link.href);
        });

        for (const apiLink of apiReferenceLinks.slice(0, 20)) { // Limit to prevent excessive requests
            try {
                console.log(`Extracting API reference: ${apiLink.text}`);
                await page.goto(apiLink.href, { waitUntil: 'networkidle2', timeout: 30000 });
                await new Promise(resolve => setTimeout(resolve, 1000));

                const apiContent = await page.evaluate(() => ({
                    title: document.title,
                    url: window.location.href,
                    content: document.body.innerText || ''
                }));

                allContent[`API_${apiLink.text}`] = apiContent;
            } catch (error) {
                console.error(`Error extracting API reference ${apiLink.href}:`, error.message);
            }
        }

    } catch (error) {
        console.error('Main scraping error:', error);
    } finally {
        await browser.close();
    }

    // Save all extracted content
    const output = {
        scrapedAt: new Date().toISOString(),
        totalPages: Object.keys(allContent).length,
        content: allContent,
        summary: {
            totalCharacters: Object.values(allContent).reduce((sum, page) => sum + (page.mainContent?.length || 0), 0),
            pages: Object.keys(allContent)
        }
    };

    fs.writeFileSync('deriv-api-docs.json', JSON.stringify(output, null, 2));
    
    // Also create a readable text version
    let readableContent = `# DERIV API DOCUMENTATION - COMPLETE EXTRACTION\n`;
    readableContent += `Scraped on: ${output.scrapedAt}\n`;
    readableContent += `Total pages: ${output.totalPages}\n\n`;

    Object.entries(allContent).forEach(([pageName, pageData]) => {
        readableContent += `\n${'='.repeat(80)}\n`;
        readableContent += `PAGE: ${pageName}\n`;
        readableContent += `URL: ${pageData.url}\n`;
        readableContent += `${'='.repeat(80)}\n\n`;
        
        if (pageData.headings?.length) {
            readableContent += `## HEADINGS:\n`;
            pageData.headings.forEach(h => {
                readableContent += `${h.level.toUpperCase()}: ${h.text}\n`;
            });
            readableContent += `\n`;
        }
        
        if (pageData.codeBlocks?.length) {
            readableContent += `## CODE EXAMPLES:\n`;
            pageData.codeBlocks.forEach((code, i) => {
                readableContent += `\n### Code Block ${i + 1} (${code.language}):\n`;
                readableContent += `\`\`\`${code.language}\n${code.content}\n\`\`\`\n`;
            });
            readableContent += `\n`;
        }
        
        readableContent += `## MAIN CONTENT:\n${pageData.mainContent}\n\n`;
    });

    fs.writeFileSync('deriv-api-docs-readable.txt', readableContent);

    console.log(`\n✅ EXTRACTION COMPLETE!`);
    console.log(`📁 Files created:`);
    console.log(`   - deriv-api-docs.json (${Math.round(fs.statSync('deriv-api-docs.json').size / 1024)}KB)`);
    console.log(`   - deriv-api-docs-readable.txt (${Math.round(fs.statSync('deriv-api-docs-readable.txt').size / 1024)}KB)`);
    console.log(`📊 Total content extracted: ${output.summary.totalCharacters.toLocaleString()} characters`);
    console.log(`📄 Pages processed: ${output.totalPages}`);

    return output;
}

// Run the scraper
scrapeDerivDocs().catch(console.error);