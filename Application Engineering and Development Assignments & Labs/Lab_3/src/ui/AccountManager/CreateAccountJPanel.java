/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JPanel.java to edit this template
 */
package ui.AccountManager;

import java.awt.CardLayout;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import model.Account;
import model.AccountDirectory;

/**
 *
 * @author tawde
 */
public class CreateAccountJPanel extends javax.swing.JPanel {
    JPanel userProcessContainer;
    AccountDirectory accountDirectory;
    /**
     * Creates new form CreateAccountJPanel
     */
    public CreateAccountJPanel(JPanel container, AccountDirectory directory) {
        initComponents();
        
        userProcessContainer = container;
        accountDirectory = directory;
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        btnBack = new javax.swing.JButton();
        lblTitle = new javax.swing.JLabel();
        lblRoutingNumber = new javax.swing.JLabel();
        lblAccountNumber = new javax.swing.JLabel();
        lblBankName = new javax.swing.JLabel();
        lblBalance = new javax.swing.JLabel();
        txtAccountNumber = new javax.swing.JTextField();
        txtBankName = new javax.swing.JTextField();
        txtRoutingNumber = new javax.swing.JTextField();
        txtBalance = new javax.swing.JTextField();
        btnCreate = new javax.swing.JButton();

        btnBack.setText("<<<Back");
        btnBack.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnBackActionPerformed(evt);
            }
        });

        lblTitle.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        lblTitle.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        lblTitle.setText("Create Account");

        lblRoutingNumber.setText("Routing Number");

        lblAccountNumber.setText("Account Number");

        lblBankName.setText("Bank Name");

        lblBalance.setText("Balance");

        btnCreate.setText("Create Account");
        btnCreate.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnCreateActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(btnBack)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(lblTitle, javax.swing.GroupLayout.PREFERRED_SIZE, 487, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(155, 155, 155)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(lblBankName)
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(lblAccountNumber)
                                    .addComponent(lblRoutingNumber)
                                    .addComponent(lblBalance))
                                .addGap(122, 122, 122)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(txtBalance, javax.swing.GroupLayout.PREFERRED_SIZE, 125, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(txtBankName, javax.swing.GroupLayout.PREFERRED_SIZE, 125, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(txtAccountNumber, javax.swing.GroupLayout.PREFERRED_SIZE, 125, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(txtRoutingNumber, javax.swing.GroupLayout.PREFERRED_SIZE, 125, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(btnCreate))))))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(btnBack)
                    .addComponent(lblTitle))
                .addGap(53, 53, 53)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblRoutingNumber)
                    .addComponent(txtRoutingNumber, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblAccountNumber)
                    .addComponent(txtAccountNumber, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblBankName)
                    .addComponent(txtBankName, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(lblBalance)
                    .addComponent(txtBalance, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(34, 34, 34)
                .addComponent(btnCreate)
                .addContainerGap(128, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void btnBackActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnBackActionPerformed
        // TODO add your handling code here:
        userProcessContainer.remove(this);
        CardLayout layout = (CardLayout) userProcessContainer.getLayout();
        layout.previous(userProcessContainer);
    }//GEN-LAST:event_btnBackActionPerformed

    private void btnCreateActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnCreateActionPerformed
        // TODO add your handling code here:
        int balance;
        
        String routingNumber = txtRoutingNumber.getText();
        String accountNumber = txtAccountNumber.getText();
        String bankName = txtBankName.getText();
        
        if (routingNumber.isBlank() || bankName.isBlank() || accountNumber.isBlank())
        {
            JOptionPane.showMessageDialog(this, "All fields are mandatory", "Error", JOptionPane.ERROR_MESSAGE);
            return;         
        }
        
        try {
            balance = Integer.parseInt(txtBalance.getText());
        } catch (Exception e){
            JOptionPane.showMessageDialog(null, "Please check the balance input", "Warning", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        Account a = accountDirectory.addAccount();
        
        a.setRoutingNumber(routingNumber);
        a.setAccountNumber(accountNumber);
        a.setBankName(bankName);
        a.setBalance(balance);
        
        JOptionPane.showMessageDialog(this, "Account successfully created", "Information", JOptionPane.INFORMATION_MESSAGE);
        
        txtRoutingNumber.setText("");
        txtAccountNumber.setText("");
        txtBankName.setText("");
        txtBalance.setText("");
        
        
    }//GEN-LAST:event_btnCreateActionPerformed


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnBack;
    private javax.swing.JButton btnCreate;
    private javax.swing.JLabel lblAccountNumber;
    private javax.swing.JLabel lblBalance;
    private javax.swing.JLabel lblBankName;
    private javax.swing.JLabel lblRoutingNumber;
    private javax.swing.JLabel lblTitle;
    private javax.swing.JTextField txtAccountNumber;
    private javax.swing.JTextField txtBalance;
    private javax.swing.JTextField txtBankName;
    private javax.swing.JTextField txtRoutingNumber;
    // End of variables declaration//GEN-END:variables
}
